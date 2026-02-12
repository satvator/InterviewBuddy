import streamlit as st
from groq import Groq
import re
import io
from streamlit_mic_recorder import mic_recorder
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. CONFIG & CLIENT ---
st.set_page_config(page_title="Interview Buddy", page_icon="🎤")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# --- 2. CORE FUNCTIONS ---
def transcribe_audio(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "input.wav"
    return client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="text"
    )

def count_fillers(text):
    fillers = ["um", "uh", "like", "actually", "basically", "you know"]
    counts = {w: len(re.findall(rf"\b{w}\b", text.lower())) for w in fillers}
    return sum(counts.values())

# --- 3. UI LAYOUT ---
st.title("🎤 Interview Buddy")
role = st.selectbox("Choose your target role:", ["Data Science", "Marketing", "Software Engineering"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = None

# --- 4. THE INTERVIEW CYCLE ---
if not st.session_state.current_q:
    if st.button("🚀 Start Interview"):
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Ask a first interview question for {role}."}],
            model=MODEL
        )
        st.session_state.current_q = resp.choices[0].message.content
        st.rerun()

if st.session_state.current_q:
    st.chat_message("assistant").write(st.session_state.current_q)
    
    # Recording Section
    audio = mic_recorder(start_prompt="⏺️ Record Answer", stop_prompt="⏹️ Stop Recording", key='recorder')

    if audio:
        with st.spinner("AI is listening..."):
            transcript = transcribe_audio(audio['bytes'])
            fillers = count_fillers(transcript)
            
            # --- 5. THE FEEDBACK HUB (With the requested Popover/Dialogue) ---
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Filler Words Used", fillers, help="Lower is better! Aims for < 3 per answer.")
            
            with col2:
                # This is the "Dialogue Box" popover you asked for
                with st.popover("ℹ️ What is this?"):
                    st.markdown("""
                    ### How to read your stats:
                    - **Filler Words:** Tracks words like 'um' and 'like' that distract listeners.
                    - **Transcript:** Review your exact words to check for pronunciation or flow issues.
                    - **Follow-up:** AI generates the next question based on your specific response.
                    """)

            with st.expander("📝 View Transcript"):
                st.write(transcript)

            # Generate next question
            st.session_state.chat_history.append({"q": st.session_state.current_q, "a": transcript})
            prompt = f"Previous context: {st.session_state.chat_history}. Ask a follow-up question for {role}."
            
            next_resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL
            )
            st.session_state.current_q = next_resp.choices[0].message.content
            
            if st.button("Continue to Next Question ➡️"):
                st.rerun()