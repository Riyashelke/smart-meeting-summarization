# smart-meeting-summarization
Smart Meeting Summarization and Action Item Detection is an AI-based system that converts meeting recordings into concise summaries and extracts tasks, deadlines, and assignees. Using Whisper, BART, PEGASUS, and BERT, it improves productivity by automating meeting analysis and prioritizing action items efficiently.

Smart Meeting Summarization and Action Item Detection

1.Overview

Smart Meeting Summarization and Action Item Detection is an AI-powered system designed to automate meeting analysis by generating concise summaries and extracting important action items from virtual and hybrid meetings. The system converts meeting recordings into structured textual insights using advanced Artificial Intelligence, Natural Language Processing (NLP), Speech Recognition, and Deep Learning techniques. It integrates OpenAI Whisper for speech transcription, BART and PEGASUS for abstractive summarization, and BERT for entity extraction and action-item detection.

The project aims to reduce manual note-taking efforts, improve productivity, enhance collaboration, and support faster decision-making in professional, educational, and industrial environments.

2.Features

Automatic Speech Recognition using Whisper
Meeting transcription from audio/video recordings
Abstractive meeting summarization
Action item extraction
Deadline and assignee detection
Task prioritization system
Multi-speaker meeting handling
Long transcript summarization
Interactive web interface
AI-powered meeting analytics

3.Problem Statement

Traditional meeting documentation methods are time-consuming and often fail to capture important discussions, decisions, and tasks accurately. Existing summarization systems struggle with multi-speaker conversations, overlapping speech, noisy audio, and long unstructured discussions. This project addresses these challenges by developing an intelligent automated system capable of generating accurate summaries and extracting actionable tasks with priorities and deadlines.

4.Objectives

Convert meeting audio into accurate text transcripts
Generate concise abstractive summaries
Extract action items automatically
Detect deadlines and assignees
Prioritize tasks based on urgency
Reduce manual meeting documentation effort
Improve productivity and meeting management

5.System Architecture

The system architecture consists of multiple processing layers:

Audio/Video Input Layer
Audio Extraction Layer
Speech Recognition Layer
NLP Summarization Layer
Action Item Detection Layer
Entity Extraction Layer
Task Prioritization Layer
Streamlit Web Interface

The architecture integrates Whisper ASR, BART, PEGASUS, and BERT models to generate structured meeting insights.

6.Methodology

The proposed system operates through a multi-stage AI pipeline:

Input meeting audio/video recordings
Extract audio using MoviePy
Convert speech into text using Whisper ASR
Generate summaries using BART and PEGASUS
Evaluate summary quality
Extract action items using BERT and rule-based techniques
Detect entities such as action, assignee, and deadlines
Prioritize tasks using scoring mechanisms
Generate final structured output

7.Algorithm

Step 1: Input Video Processing
Load MP4 meeting recordings
Extract audio using MoviePy
Convert audio into MP3 format

Step 2: Speech Recognition
Process audio using Whisper Large-v2
Generate meeting transcripts

Step 3: Meeting Summarization
Pass transcript into BART and PEGASUS
Compare summary quality
Select best-performing summary

Step 4: Action Item Detection
Apply rule-based keyword detection
Extract ML features
Identify actionable tasks

Step 5: Entity Extraction
Use BERT with BIO tagging
Detect:
Action
Assignee
Deadline

Step 6: Task Prioritization
Assign urgency scores
Rank tasks based on contextual importance

Step 7: Final Output
Generate summaries
Generate action-item list
Generate priority-based tasks

8.Dataset Description

AMI Meeting Corpus

The project uses the AMI Meeting Corpus dataset containing approximately 140 multi-speaker meetings recorded in English.

Dataset Includes:
Audio recordings
Video recordings
Meeting transcripts
XML annotations
Speaker interactions
Supported Formats:
MP4
WAV
XML
RTF

The dataset is suitable for meeting summarization, speech recognition, and action-item extraction research.

9.Technologies Used

Programming Language
Python

Libraries & Frameworks
Transformers
Hugging Face
PyTorch
MoviePy
NumPy
Pandas
NLTK
spaCy
Streamlit

10.Models Used

Whisper:

OpenAI Whisper is used for Automatic Speech Recognition (ASR). It converts meeting audio into text transcripts while handling noisy environments and multi-speaker conversations.

BART:

BART is a transformer-based abstractive summarization model that generates concise summaries from long conversational transcripts.

PEGASUS:

PEGASUS is designed specifically for text summarization tasks and produces highly coherent and context-aware summaries.

BERT:

BERT with BIO tagging is used for entity extraction and action-item detection. It identifies tasks, deadlines, and assigned persons.

11.Installation

Clone Repository
git clone https://github.com/Riyashelke/smart-meeting-summarization.git

Move into Project Directory
cd smart-meeting-summarization

Create Virtual Environment
python -m venv venv

12.Activate Environment:

macOS/Linux
source venv/bin/activate

Windows
venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

13.Usage
Run Main Pipeline
python pipeline_combined.py

Run API
python pipeline_api.py

13.Expected Output

The system generates:

Meeting transcript
Abstractive summary
Action items
Assignee information
Deadlines
Priority-ranked task list

14.Results and Analysis

The project compares the performance of BART and PEGASUS models based on:

Processing time
Summary quality
Semantic coherence
Information retention
Context understanding

Observations:
PEGASUS achieved better summary quality (~0.84)
BART achieved summary quality (~0.78)
PEGASUS demonstrated more stable processing performance
The hybrid action-item detection system achieved over 90% accuracy in tested meeting cases

15.Applications
Corporate meeting analysis
Online collaboration platforms
Educational discussions
Industrial workflow management
Productivity enhancement systems
AI-powered documentation tools

16.Advantages
Reduces manual note-taking
Saves time
Improves productivity
Generates structured insights
Supports long meeting analysis
Handles multi-speaker conversations
Automates task extraction

17.Future Enhancements
Real-time meeting summarization
Multilingual support
Cloud deployment
Advanced speaker diarization
Live meeting assistant integration
Sentiment analysis dashboard
Mobile application support

18.Conclusion

The Smart Meeting Summarization and Action Item Detection system successfully automates the process of meeting transcription, summarization, and task extraction using AI and NLP technologies. By integrating Whisper, BART, PEGASUS, and BERT models, the system generates accurate summaries, extracts actionable insights, and prioritizes tasks effectively. The project improves productivity, reduces manual effort, and provides a scalable solution for modern meeting management.

19.Authors
Riya Shelke
Vaishnavi Ranaware
Namoha Goyal
Dr. Pranali Kosamkar (Guide)

MIT World Peace University

20.References
AMI Meeting Corpus
OpenAI Whisper
Hugging Face Transformers
BART Research Paper
PEGASUS Research Paper
BERT Research Paper
QMSum Dataset
SAMSum Dataset

License
This project is developed for academic and research purposes.

GitHub Repository
Repository Link:
https://github.com/Riyashelke/smart-meeting-summarization
