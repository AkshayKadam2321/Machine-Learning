import streamlit as st
import pickle
import numpy as np

# Load Pickle Files
books = pickle.load(open(r'C:\Users\inter\Desktop\Shrutika\ML Project\Book Recomendation(Collabroative Clustering)\books.pkl', 'rb'))
pivot_table = pickle.load(open(r'C:\Users\inter\Desktop\Shrutika\ML Project\Book Recomendation(Collabroative Clustering)\pivot_table.pkl', 'rb'))
similarity_scores = pickle.load(open(r'C:\Users\inter\Desktop\Shrutika\ML Project\Book Recomendation(Collabroative Clustering)\similarity_scores.pkl', 'rb'))

# Recommendation Function
def recommend(book_name):

    if book_name not in pivot_table.index:
        return []

    index = np.where(
        pivot_table.index == book_name
    )[0][0]

    similar_books = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for i in similar_books:

        temp_df = books[
            books['Book-Title'] ==
            pivot_table.index[i[0]]
        ]

        book_details = temp_df.drop_duplicates(
            'Book-Title'
        )[[
            'Book-Title',
            'Book-Author',
            'Publisher',
            'Year-Of-Publication'
        ]]

        recommendations.append(
            book_details.values.tolist()[0]
        )

    return recommendations


# Streamlit UI
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚"
)

st.title("📚 Book Recommendation System")

selected_book = st.selectbox(
    "Select a Book",
    pivot_table.index.values
)

if st.button("Recommend Books"):

    recommendations = recommend(selected_book)

    if len(recommendations) == 0:
        st.error("Book not found in dataset.")
    else:
        st.success("Recommended Books")

        for idx, rec in enumerate(recommendations, start=1):

            st.subheader(f"Recommendation {idx}")

            st.write(f"**Title:** {rec[0]}")
            st.write(f"**Author:** {rec[1]}")
            st.write(f"**Publisher:** {rec[2]}")
            st.write(f"**Year:** {rec[3]}")

            st.markdown("---")