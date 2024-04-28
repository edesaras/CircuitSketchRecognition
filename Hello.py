import streamlit as st

if __name__ == "__main__":
    # set page configurations and display/annotation options
    st.set_page_config(
        page_title="Circuit Sketch Recognizer",
        layout="wide"
    )

    st.title("Circuit Sketch Recognition")
    col1, col2 = st.columns(2)
    with col1:
        st.image('example1.jpg', use_column_width=True, caption='Example 1')
    with col2:
        st.image('example2.jpg', use_column_width=True, caption='Example 2')
