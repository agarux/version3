import os
 
# Function to rename multiple files
def main():
   
    folder = "./data/cameras/000442922112/depth_frames/"
    count=1
    for filename in sorted(os.listdir(folder)):
        print(count)
        dst = f"000442922112_depth_f{count:04d}.png"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
        count += 1 
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()