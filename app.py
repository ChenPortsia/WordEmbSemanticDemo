import streamlit as st
from visualizer import SemanticSpaceVisualizer
from model_loader import ModelLoader
from utils import validate_input, parse_input

@st.cache_resource
def get_model_loader():
    return ModelLoader()

def main():
    st.title("Semantic Space Visualizer")

    model_loader = get_model_loader()
    visualizer = SemanticSpaceVisualizer()

    # Model selection
    available_models = ['word2vec', 'glove', 'fasttext']
    selected_models = st.multiselect("Select models to compare", available_models, default=['word2vec'])

    # Input fields
    x_space_base = st.text_input("X-axis base words (comma-separated)", "expensive")
    x_space_contra = st.text_input("X-axis contrast words (comma-separated)", "cheap")
    y_space_base = st.text_input("Y-axis base words (comma-separated)", "big")
    y_space_contra = st.text_input("Y-axis contrast words (comma-separated)", "small")

    group1 = st.text_input("Group 1 words (comma-separated)", "dog, cat, bird, fish")
    group2 = st.text_input("Group 2 words (comma-separated)", "car, bicycle, motorcycle, bus")
    group3 = st.text_input("Group 3 words (comma-separated)", "apple, banana, orange, pear")

    operation = st.selectbox("Select operation (optional)", ['None', 'add', 'subtract', 'multiply', 'divide', 'average'])
    target_group = st.selectbox("Select target group for operation", ['all', 'group_1', 'group_2', 'group_3'])
    extra_word = st.text_input("Extra word for operation (if applicable)")

    if st.button("Visualize"):
        # Parse input
        x_space = (parse_input(x_space_base), parse_input(x_space_contra))
        y_space = (parse_input(y_space_base), parse_input(y_space_contra))
        groups = [parse_input(group1), parse_input(group2), parse_input(group3)]
        operation = None if operation == 'None' else operation

        # Validate input
        is_valid, error_message = validate_input(x_space, y_space, groups, operation, target_group, extra_word)

        if is_valid:
            for model_name in selected_models:
                model = model_loader.load_model(model_name)
                x_coords, y_coords = visualizer.semantic_space_2d_representation(model, x_space, y_space, groups, operation, target_group, extra_word)
                fig = visualizer.plot_semantic_space_2d(x_coords, y_coords, groups, x_space, y_space)
                st.pyplot(fig)
        else:
            st.error(error_message)

if __name__ == "__main__":
    main()