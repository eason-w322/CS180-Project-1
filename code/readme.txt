This project is organized under the CS180-Project-1 folder. Inside, you’ll find:
code/main.py → the entry point (this is what you run)
code/utils.py → helper functions
README.txt → this file
data/ → (intentionally left blank for you to import images to this folder)
website → a pdf of my webpage

Data not included: I did not submit the raw .tif images (too large / not required). 
To test the code, place your .tif images in a folder named data/ at the root of the project.

By default, the script looks for input under data/ (e.g., data/image.tif)
It saves the aligned output under results/ (e.g., results/image_rgb.jpg).
To test a different image, edit the bottom of main.py(the image needs to be imported first):
in_file  = ROOT / "data" / "your_image.tif"
out_file = ROOT / "results" / "your_image_rgb.jpg"

Dependencies: numpy, scikit-image  
(You can install them with `pip install -r requirements.txt` if you create a requirements file.)

To run the program:
cd CS180-Project-1 and run:
python3 code/main.py

A new results/ folder will be created automatically, and the processed .jpg will be saved there.
THANK YOU!!

