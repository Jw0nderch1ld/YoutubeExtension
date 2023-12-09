from flask import Flask, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)


model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


@app.get("/summary")
def summary_api():
    url = request.args.get("url", "")
    video_id = url.split("=")[1]

    transcript = get_transcript(video_id)
    if not transcript:
        return {"error": "Failed to retrieve transcript"}, 404

    try:
        summary = get_summary(transcript)
        return {"summary": summary}, 200
    except IndexError:
        return {"error": "Failed to generate summary"}, 500


def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en"], preserve_formatting=True
        )
        transcript = " ".join([d["text"] for d in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def get_summary(transcript):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    summary = ""
    for i in range(0, (len(transcript) // 1000) + 1):
        summary_text = summarizer(transcript[i * 1000 : (i + 1) * 1000])[0][
            "summary_text"
        ]
        summary = summary + summary_text + " "
    return summary


if __name__ == "__main__":
    app.run()
