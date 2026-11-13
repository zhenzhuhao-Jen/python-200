from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key")
client = OpenAI()


# Task 1: Setup and System Prompt
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", 
    "content": "You are a a job application coach who will help a user, who is changing their career, rewrite their resume bullet points, generate a draft cover letter, and ask follow-up questions. help the user to brainstorm ways to map their previous experience to the skills their new job requires, Stay focused on job application materials. Always remind the user to review and edit its output before submitting anywhere. Acknowledge that you may not know the user's specific industry norms, and that the user should use their own judgment "}
]

# I added the backgound for the user, because it's close to our background.



# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).don't add the word "json" outside of the JSON list. 

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    
    try:
        result = json.loads(response)
        for i in result:
            print(f"original: {i['original']}, improved: {i['improved']}")     
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(bullets)
# The original bullets are short and vague. The model made the context more specific and strong.



# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    return  get_completion(messages)
    

result = generate_cover_letter(job_title = "Junior Data Engineer", background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Pandas.")
print(result)
# choosing particular examples help the model understand the style and tone the user want. the few_shot pattern will ensure the user is getting what they want specificly.


# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("The input is flagged as unsafe.")
        return False   
    else:
        return True

test1 = is_safe("I have an apple and a orange!")
print(test1)
test2 = is_safe("I have a knife and I want to cut a finger.")
print(test2)



#Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", 
        "content": "You are a a job application coach who will help a user, who is changing their career, rewrite their resume bullet points, generate a draft cover letter, and ask follow-up questions. help the user to brainstorm ways to map their previous experience to the skills their new job requires, Stay focused on job application materials. Always remind the user to review and edit its output before submitting anywhere. Acknowledge that you may not know the user's specific industry norms, and that the user should use their own judgment "}
    ]
  

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewrite_bullets(raw_bullets)
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            print(generate_cover_letter(job_title, background))
        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            result = get_completion(messages)
            print(result)
            messages.append({"role": "assistant", "content": result})

            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            pass
        


if __name__ == "__main__":
    run_chatbot()



# Task 6: Ethics Reflection

# I choose Option A — Comment block
# 2. if a job-seeker submitted the bot's output directly — without reviewing it, the possible unrelated information could pass to the real employer and cause confusion.
# 3. If I were deploying this tool professionally, I would add a guardrail disclaimer about using the output. remind the user to review and edit the output before submitting.
