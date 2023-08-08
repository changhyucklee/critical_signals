
from sharepoint import SharePoint

#i.e - file_dir_path = r'C:\project\report.pdf'
file_dir_path = r'D:\Python\DecoaterFeedRate\plots\2023-03-07\2023-03-07_RT_1.png'

# this will be the file name that it will be saved in SharePoint as 
file_name = '2023-03-07_RT_1.png'

# The folder in SharePoint that it will be saved under
folder_name = '2023-03-07'

# upload file
SharePoint().upload_file(file_dir_path, file_name, folder_name)

# delete file
# SharePoint().delete_file(file_name, folder_name)
