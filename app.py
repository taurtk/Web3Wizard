import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate

def generate_tweets(api_key, tweets):
    """Generate tweets using Groq LLM"""
    try:
        # Set the API key
        os.environ["GROQ_API_KEY"] = api_key

        # Initialize the Groq LLM
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=1,
            max_tokens=1024,
            top_p=1,
        )

        # Create a detailed prompt template to generate tweets
        template = """You are Web3Wizard, a crypto and tech-focused Twitter personality. 
        Based on these example tweets, generate 10 new tweets that capture the style, tone, and themes:

        Example Tweets:
        {example_tweets}

        Guidelines for generated tweets:
        - Use crypto and tech-related language
        - Include emojis sparingly
        - Focus on investment insights
        - Use short, punchy sentences
        - Incorporate cryptocurrency ticker symbols
        - Maintain an energetic and confident tone

        Generate 10 unique tweets that sound like they could be from Web3Wizard. 
        Format each tweet on a new line, starting with "Tweet X:":"""

        # Create prompt with tweet examples
        prompt = PromptTemplate(
            input_variables=["example_tweets"],
            template=template,
        )

        # Create an LLMChain for tweet generation
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Generate tweets
        response = llm_chain.run(example_tweets="\n".join(tweets))
        st.write("Raw LLM Response:", response)  # Debug print

        # Extract tweets using regex
        tweet_pattern = r'Tweet \d+:\s*(.+?)(?=Tweet \d+:|$)'
        generated_tweets = re.findall(tweet_pattern, response, re.DOTALL)
        st.write("Extracted Tweets:", generated_tweets)  # Debug print

        # Clean and truncate tweets
        generated_tweets = [
            tweet.strip()[:280] for tweet in generated_tweets 
            if tweet.strip()
        ][:10]

        return generated_tweets

    except Exception as e:
        st.error(f"Error in generate_tweets: {str(e)}")
        return []

def main():
    # Set page config
    st.set_page_config(page_title="Web3Wizard Tweet Generator", page_icon="🚀")

    # Title and description
    st.title("🚀 Web3Wizard Tweet Generator")
    st.write("Generate crypto-focused tweets in the style of Web3Wizard")

    # Groq API Key input
    api_key = st.text_input("Enter your Groq API Key", type="password")

    # Default tweets
    default_tweets = [
        "Mf*<krs always find $ to invest when the market is \"UP ONLY\" & seemingly easy money is on the table!",
        "I 👁️ a lot more people waking up to $TAO",
        "$TAO broke resistance of the downwards trend line.",
        "You have 2 CHOICES: 1. Position yourself before the PUMPS = WIN 2. Chase the PUMPS = LOSE",
        "An easy $100M A.I. agent infra project sitting at $25M & you're messing about with single agents?",
        "$ETH & the gas fees — sort it out",
        "LFG $BTC We did $100k! Currently $103k - it was inevitable.",
        "It's ONLY EASY NOW because you took the hard route previously.",
        "AI - AI Agents are the play.",
        "ALL INVESTMENTS are just vehicles to wealth."
    ]

    # Textarea for custom tweets
    st.subheader("Training Tweets")
    custom_tweets = st.text_area(
        "Customize training tweets (one per line)", 
        value="\n".join(default_tweets), 
        height=200
    )

    # Generate button
    if st.button("Generate Tweets"):
        # Validate inputs
        if not api_key:
            st.error("Please enter a Groq API Key")
            return
        
        # Split custom tweets
        tweets_list = [tweet.strip() for tweet in custom_tweets.split('\n') if tweet.strip()]

        try:
            # Generate tweets
            generated_tweets = generate_tweets(api_key, tweets_list)

            # Display generated tweets
            if generated_tweets:
                st.subheader("Generated Tweets:")
                for i, tweet in enumerate(generated_tweets, 1):
                    st.markdown(f"**Tweet {i}:** {tweet}")
            else:
                st.warning("No tweets were generated. Please check the API key and try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info(
        "This app generates tweets in the style of Web3Wizard using Groq's LLaMA 3 model. "
        "Customize the training tweets to influence the generated content."
    )

if __name__ == "__main__":
    main()