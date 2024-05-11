import streamlit as st
import dropbox
import os
import hmac

# Initialize the Dropbox client
dbx = dropbox.Dropbox(os.getenv("DROPBOX_ACCESS_TOKEN"))

def get_text_files(path):
    """Retrieve all text files from a specified Dropbox folder path."""
    try:
        files = dbx.files_list_folder(path).entries
        return [f.name for f in files if isinstance(f, dropbox.files.FileMetadata) and f.name.endswith('.txt')]
    except dropbox.exceptions.ApiError as e:
        st.error("Failed to fetch files: {}".format(e))
        return []

def read_file(path):
    """Read the content of a text file from Dropbox."""
    try:
        _, res = dbx.files_download(path)
        return res.content.decode("utf-8")
    except dropbox.exceptions.ApiError as e:
        st.error("Failed to download file: {}".format(e))
        return ""

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


def main():
    st.sidebar.title("Text Files")
    # Specify the path to the folder in your Dropbox
    folder_path = "/Apps/NewBizBot/"

    # Get the list of text files
    file_names = get_text_files(folder_path)
    selected_file = st.sidebar.selectbox("Select a file", file_names)

    if selected_file:
        # Display the content of the selected file
        file_path = os.path.join(folder_path, selected_file)
        file_content = read_file(file_path)
        st.markdown(file_content)

if __name__ == "__main__":
    main()
