from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class GoogleConnet:

    @staticmethod
    def connect():
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        file_list = drive.ListFile().GetList()
        for file in file_list:
            print("Title: %s, ID: %s" % (file['title'], file['id']))
            if file['title'] == "To Share":
                file_id = file['id']
