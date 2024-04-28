import streamlit as st

if __name__ == "__main__":
    # set page configurations and display/annotation options
    st.set_page_config(
        page_title="Circuit Sketch Recognizer",
        layout="wide"
    )
    st.title("Circuit Sketch Recognition")
    st.markdown(
        '''
        Can computers recognize hand-drawn circuit sketches? Absolutely!

        Upload a picture of your circuit sketch or take a new one to see it in action.

        This project leverages fine tuned TrOCR for text recognition and YOLOv8 for component detection, using CGHD-2304 dataset. 
        '''
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image('media/capture.gif', use_column_width=True, caption='Take a Picture')
    with col2:
        st.image('media/upload.gif', use_column_width=True, caption='Upload an Image')
    st.markdown('Here are some more examples!')
    col3, col4 = st.columns(2)
    with col3:
        st.image('media/example1.jpg', use_column_width=True, caption='Example 1')
    with col4:
        st.image('media/example2.jpg', use_column_width=True, caption='Example 2')
