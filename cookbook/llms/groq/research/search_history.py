import streamlit as st
import os
import hmac
import boto3
from botocore.client import Config


# Initialize the S3 client for Cloudflare R2 storage
endpoint_url = 'https://44ae5977e790e0a48e71df40637d166a.r2.cloudflarestorage.com/'
access_key = os.getenv('CLOUDFLARE_ACCESS_KEY')
secret_key = os.getenv('CLOUDFLARE_SECRET_KEY')
bucket_name = 'newbizbot'  # Your hardcoded bucket name

s3_client = boto3.client('s3',
                         region_name='auto',
                         endpoint_url=endpoint_url,
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key,
                         config=Config(signature_version='s3v4'))

def get_text_files(folder_path):
    """Retrieve all text files from a specified folder in Cloudflare R2 Storage."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        files = response.get('Contents', [])
        return [item['Key'] for item in files if item['Key'].endswith('.txt')]
    except s3_client.exceptions.ClientError as e:
        print(f"Failed to fetch files: {e}")
        return []

def read_file(file_key):
    """Read the content of a text file from Cloudflare R2 Storage."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return response['Body'].read().decode('utf-8')
    except s3_client.exceptions.ClientError as e:
        print(f"Failed to download file: {e}")
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
