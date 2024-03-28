import os 
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_openai import OpenAI
from langchain.chains import ConversationChain 
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from whisper import load_model


whisper_model = load_model("base")

load_dotenv()


app = App(token=os.environ.get('SLACK_BOT_TOKEN'))

#Langchain implementation
template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {human_input}
    Assistant:"""

prompt = PromptTemplate(
    input_variables=["history","human_input"],
    template=template
)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2)
)

@app.event("file_shared")
def handle_file_shared(event, client, logger):
    try:
        # Initial setup remains the same, you download the shared file
        file_id = event.get("file_id")
        result = client.files_info(file=file_id)
        file_info = result.get("file", {})
        file_url_private = file_info.get("url_private_download")

        headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
        response = requests.get(file_url_private, headers=headers, stream=True)

        if response.status_code == 200:
            video_path = 'downloaded_video.mp4'
            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract audio and transcribe it
            audio_path = 'audio.wav'
            os.system(f"ffmpeg -y -i {video_path} -acodec pcm_s16le -ar 16000 {audio_path}")

            result = whisper_model.transcribe(audio_path)
            transcription = result["text"]

            # Here's the new part: Generate an OpenAI response using the transcription
            output = chatgpt_chain.predict(human_input=transcription)
            
            # Post the OpenAI-generated response instead of the transcription
            channel_id = event.get("channel_id")
            client.chat_postMessage(channel=channel_id, text=output)

        else:
            logger.error(f"Failed to download video file. Status code: {response.status_code}")

    except Exception as e:
        logger.error(f"Error handling file_shared event: {e}")





@app.message(".*")
def message_handler(message,say,logger):
    print(message)

    output = chatgpt_chain.predict(human_input = message['text'])
    say(output)

if __name__ == "__main__":
    SocketModeHandler(app,os.environ["SLACK_APP_TOKEN"]).start()
