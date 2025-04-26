import os
import re
import yt_dlp as youtube_dl
import time
import whisper
from datetime import timedelta
import pandas as pd
from nltk.tokenize import sent_tokenize
import streamlit as st

class YouTubeTranscriptAgent:
    def __init__(self, model_size="base"):
        """
        Initialize the YouTube Transcript Agent.
        
        Args:
            model_size (str): Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model = whisper.load_model(model_size)
        self.transcript_df = None
        self.video_info = {}
        
    def download_audio(self, youtube_url, output_dir=None):
        """
        Download audio from a YouTube video using yt-dlp.
        
        Args:
            youtube_url (str): URL of the YouTube video
            output_dir (str): Directory to save the downloaded audio
            
        Returns:
            str: Path to the downloaded audio file
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "audio")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video ID from the URL
        video_id = None
        if "youtube.com/watch" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = "video"  # Fallback name
            
        output_file = os.path.join(output_dir, f"{video_id}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        # Download video info first
        info_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with youtube_dl.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Store video information
                self.video_info = {
                    'title': info.get('title', 'Unknown'),
                    'author': info.get('uploader', 'Unknown'),
                    'publish_date': info.get('upload_date', 'Unknown'),
                    'views': info.get('view_count', 0),
                    'length': info.get('duration', 0),
                    'url': youtube_url,
                    'thumbnail_url': info.get('thumbnail', '')
                }
            
            # Now download the audio
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                
            # The actual filename might be different because of yt-dlp's processing
            output_file = os.path.join(output_dir, f"{video_id}.mp3")
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Transcription result
        """
        result = self.model.transcribe(audio_path, fp16=False)
        
        # Create DataFrame from segments
        data = []
        for segment in result['segments']:
            data.append({
                'start': segment['start'],
                'end': segment['end'],
                'start_str': str(timedelta(seconds=int(segment['start']))),
                'text': segment['text'].strip()
            })
        
        self.transcript_df = pd.DataFrame(data)
        return result
    
    def search_transcript(self, query):
        """
        Search for a query in the transcript.
        
        Args:
            query (str): Text to search for
            
        Returns:
            DataFrame: Matching segments
        """
        if self.transcript_df is None:
            st.error("No transcript available. Please transcribe a video first.")
            return None
        
        pattern = re.compile(query, re.IGNORECASE)
        matches = self.transcript_df[self.transcript_df['text'].str.contains(pattern)]
        
        return matches
    
    def summarize_transcript(self, num_sentences=5):
        """
        Generate a simple extractive summary of the transcript.
        
        Args:
            num_sentences (int): Number of sentences to include in summary
            
        Returns:
            str: Summary text
        """
        if self.transcript_df is None:
            st.error("No transcript available. Please transcribe a video first.")
            return None
        
        # Combine all transcript text
        full_text = " ".join(self.transcript_df['text'].tolist())
        
        # Split into sentences
        sentences = sent_tokenize(full_text)
        
        # For this simple implementation, just take the first few sentences
        # A more sophisticated approach would use TextRank or another algorithm
        if num_sentences >= len(sentences):
            summary = " ".join(sentences)
        else:
            summary = " ".join(sentences[:num_sentences])
        
        return summary
    
    def get_timestamps(self, query):
        """
        Get timestamps for segments containing the query.
        
        Args:
            query (str): Text to search for
            
        Returns:
            list: List of (timestamp, text) tuples
        """
        matches = self.search_transcript(query)
        if matches is None or len(matches) == 0:
            return []
        
        timestamps = [(row['start_str'], row['text']) for _, row in matches.iterrows()]
        return timestamps
    
    def get_full_transcript(self):
        """
        Get the full transcript as a formatted string.
        
        Returns:
            str: Formatted transcript
        """
        if self.transcript_df is None:
            return "No transcript available."
        
        transcript_text = ""
        for _, row in self.transcript_df.iterrows():
            transcript_text += f"[{row['start_str']}] {row['text']}\n"
        
        return transcript_text

def format_time(seconds):
    """Format seconds to mm:ss or hh:mm:ss."""
    td = timedelta(seconds=seconds)
    if td.seconds < 3600:
        return f"{td.seconds // 60:02d}:{td.seconds % 60:02d}"
    else:
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_youtube_embed_url(youtube_url):
    """Convert a YouTube video URL to an embedded URL."""
    video_id = None
    
    # Extract video ID from URL
    if "youtube.com/watch" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    
    if video_id:
        return f"https://www.youtube.com/embed/{video_id}"
    return None