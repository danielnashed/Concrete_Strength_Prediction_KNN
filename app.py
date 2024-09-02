from collections import namedtuple
import streamlit as st
import knn, visualizations
import time

st.set_page_config(
    layout="wide")

custom_css = """
<style>
h1 {
    margin-top: 0;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True) 

@st.cache_data
def start_session():
    # Load the data
    data = knn.parse_data("concrete_compressive_strength.csv") 
    # Get the min and max and mean values for each column 
    minmax = knn.get_min_max_mean(data) 
    # Normalize the data for visualizations
    normalized_data = knn.normalize(data, minmax)
    return data, minmax, normalized_data

def main():
    st.header('K-Nearest Neighbour Algorithm', divider='gray')
    st.write('This is a simple implementation of the K-Nearest Neighbor algorithm to predict the compressive strength of concrete in MPa.')
    k = st.slider('Choose k, the number of neighbours to consider for predictions:', 1, 20, 5) 
    data, minmax, normalized_data = start_session()

    # Create a left sidebar to choose parameters
    st.sidebar.header('Choose Parameters')
    cement = st.sidebar.slider('Cement: Kg in a cubic meter', minmax[0][0], minmax[0][1], minmax[0][2])
    slag = st.sidebar.slider('Slag: Kg in a cubic meter', minmax[1][0], minmax[1][1], minmax[1][2])
    ash = st.sidebar.slider('Ash: Kg in a cubic meter', minmax[2][0], minmax[2][1], minmax[2][2])
    water = st.sidebar.slider('Water: Kg in a cubic meter', minmax[3][0], minmax[3][1], minmax[3][2])
    superplasticizer = st.sidebar.slider('Superplasticizer: Kg in a cubic meter', minmax[4][0], minmax[4][1], minmax[4][2])
    coarse_aggregate = st.sidebar.slider('Coarse Aggregate: Kg in a cubic meter', minmax[5][0], minmax[5][1], minmax[5][2])
    fine_aggregate = st.sidebar.slider('Fine Aggregate: Kg in a cubic meter', minmax[6][0], minmax[6][1], minmax[6][2])
    age = st.sidebar.slider('Age: Days', minmax[7][0], minmax[7][1], minmax[7][2])

    # Create query point
    query = [cement, slag, ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age, 0]

    # Run inference using KNN
    start_time = time.time()
    k_nearest_neighbors = knn.knn(data, query, k) 
    prediction = knn.predict(k_nearest_neighbors)
    query[-1] = prediction
    end_time = time.time()

    # Display prediction and query response time
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'Predicted Compressive Strength: <span style="color:red; font-weight:bold; font-size:20px;">{prediction:.3f}</span> MPa', unsafe_allow_html=True)
    with col2:
        st.write(f'Query Response Time: {(end_time - start_time)*1000:.3f} ms')

    # Plot database with query point and k-nearest neighbors using parallel coordinates
    fig, ax = visualizations.plot_dataset(normalized_data)
    fig_after_inference, _ = visualizations.update_plot(fig, ax, minmax, k_nearest_neighbors, query)
    st.pyplot(fig_after_inference)

    # Display query point and k-nearest neighbors in a table
    table = visualizations.tabulate_data(query, k_nearest_neighbors)
    with st.container():
        st.markdown(table, unsafe_allow_html=True)
        st.write('')

    # Display proximity plot of query point and k-nearest neighbors
    visualizations.proximity_plot(k_nearest_neighbors)
    with st.container():
        st.image('proximity_fig.png')

    st.write('\n\n By Daniel Nashed, 2024.')

if __name__ == "__main__":
    main() 