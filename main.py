from utils import YouTubeTranscriptAgent, create_youtube_embed_url, format_time 
import streamlit as st
import nltk



# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



def main():
    st.set_page_config(
        page_title="YouTube Transcript AI Agent",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 YouTube Transcript AI Agent")
    
    # Initialize session state variables if they don't exist
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'transcript_generated' not in st.session_state:
        st.session_state.transcript_generated = False
    if 'video_info' not in st.session_state:
        st.session_state.video_info = {}
    
    # Sidebar for model selection and URL input
    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox(
            "Select Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to 'base'
            help="Larger models are more accurate but slower and require more memory"
        )
        
        youtube_url = st.text_input("YouTube Video URL", "")
        
        if st.button("Generate Transcript"):
            if youtube_url:
                try:
                    with st.spinner("Initializing Whisper model..."):
                        st.session_state.agent = YouTubeTranscriptAgent(model_size=model_size)
                    
                    with st.spinner("Downloading audio..."):
                        audio_path = st.session_state.agent.download_audio(youtube_url)
                    
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        st.session_state.agent.transcribe_audio(audio_path)
                    
                    st.session_state.transcript_generated = True
                    st.session_state.video_info = st.session_state.agent.video_info
                    st.success("Transcript generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.info("If you're experiencing download issues, make sure the video is publicly accessible and not age-restricted.")
            else:
                st.warning("Please enter a YouTube URL")
    
    # Main content area - display transcript and features when available
    if st.session_state.transcript_generated and st.session_state.agent:
        # Create two columns for video info and video player
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Video Information")
            st.write(f"**Title:** {st.session_state.video_info.get('title', 'Unknown')}")
            st.write(f"**Channel:** {st.session_state.video_info.get('author', 'Unknown')}")
            st.write(f"**Duration:** {format_time(st.session_state.video_info.get('length', 0))}")
            st.write(f"**Views:** {st.session_state.video_info.get('views', 'Unknown')}")
        
        with col2:
            # Display embedded YouTube video
            embed_url = create_youtube_embed_url(st.session_state.video_info.get('url', ''))
            if embed_url:
                st.components.v1.iframe(embed_url, height=300, width=None)
            else:
                st.image(st.session_state.video_info.get('thumbnail_url', ''))
        
        # Tabs for different features
        tab1, tab2, tab3 = st.tabs(["Full Transcript", "Search & Timestamps", "Summarize"])
        
        with tab1:
            st.subheader("Full Transcript")
            transcript_text = st.session_state.agent.get_full_transcript()
            st.text_area("", transcript_text, height=400)
            
            # Download button for transcript
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name=f"{st.session_state.video_info.get('title', 'transcript')}.txt",
                mime="text/plain"
            )
        
        with tab2:
            st.subheader("Search Transcript")
            search_query = st.text_input("Enter search term:", "")
            
            if search_query:
                matches = st.session_state.agent.search_transcript(search_query)
                
                if matches is not None and len(matches) > 0:
                    st.success(f"Found {len(matches)} matches for '{search_query}'")
                    
                    # Create a DataFrame for display
                    display_df = matches[['start_str', 'text']].copy()
                    display_df.columns = ['Timestamp', 'Text']
                    
                    # Add YouTube time links
                    video_id = None
                    if "youtube.com/watch" in st.session_state.video_info.get('url', ''):
                        video_id = st.session_state.video_info.get('url', '').split("v=")[1].split("&")[0]
                    elif "youtu.be/" in st.session_state.video_info.get('url', ''):
                        video_id = st.session_state.video_info.get('url', '').split("youtu.be/")[1].split("?")[0]
                    
                    if video_id:
                        display_df['Link'] = display_df.apply(
                            lambda row: f"[Go to timestamp](https://www.youtube.com/watch?v={video_id}&t={int(matches.loc[row.name, 'start'])}s)", 
                            axis=1
                        )
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info(f"No matches found for '{search_query}'")
        
        with tab3:
            st.subheader("Summarize Transcript")
            num_sentences = st.slider("Number of sentences in summary:", 3, 20, 5)
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.agent.summarize_transcript(num_sentences)
                    st.markdown("### Summary")
                    st.write(summary)
                    
                    # Download button for summary
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"{st.session_state.video_info.get('title', 'summary')}_summary.txt",
                        mime="text/plain"
                    )
    else:
        # Display instructions when no transcript is available
        st.info("👈 Enter a YouTube URL in the sidebar and click 'Generate Transcript' to begin.")
        
        st.markdown("""
        ### How to use this tool:
        
        1. Select a Whisper model size in the sidebar (larger models are more accurate but slower)
        2. Enter a YouTube URL in the sidebar
        3. Click 'Generate Transcript' to process the video
        4. Once processing is complete, you can:
           - View the full transcript
           - Search for specific words or phrases
           - Generate a summary of the content
        
        ### Requirements:
        - Internet connection to download the YouTube video and Whisper model
        - Sufficient memory for the selected model size
        
        ### Troubleshooting:
        - If you encounter download errors, try a different video
        - Some videos may be protected or region-restricted
        - Larger videos may require more processing time
        """)

if __name__ == "__main__":
    main()