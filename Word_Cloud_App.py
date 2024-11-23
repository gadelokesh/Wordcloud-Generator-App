# import the necessary libraries
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from wordcloud import WordCloud
from nltk.corpus import stopwords
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tree import Tree
from io import BytesIO

# title
st.title("WordCloud Generator")

# descripton
st.write('Transform your text into a vibrant word cloud with This simple APP!') 
st.write("Just paste in your words And Generate your own WORD CLOUD!")

# inputtext
text=st.text_area("Enter the Text Here")




# Entity Recogniaztion chunks
if st.button("Entity Recognization"):
    # tokenize the words
    tokens=word_tokenize(text)
    # pos_tag the tokens
    tag_token=nltk.pos_tag(tokens)
    # generate Entity chunks
    er_chunk=ne_chunk(tag_token)
    
    # see the details
    st.write("Entity Details")
    st.text(er_chunk)

# generate word cloud
if st.button("Generate Word Cloud"):
    # Generate word cloud
    wordcloud = WordCloud(width=420, height=200, margin=2,background_color='black',colormap='Accent',mode='RGBA').generate(text)    
    # Display word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="antialiased")
    plt.axis("off")
    st.pyplot(plt)
    
    # Save word cloud to an in-memory file
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    buffer.seek(0)

    # Download button for the word cloud image
    st.download_button(
        label="Download Image",
        data=buffer,
        file_name="word_cloud.png",
        mime="image/png"
    )