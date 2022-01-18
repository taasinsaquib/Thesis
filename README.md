# Thesis
All Work related to my MS Thesis

# May - June 2021
* Initial idea to work with the biomimetic eye model 
* Incorporate it into the final project for CS 275 - Artificial Life - Demetri Terzpoulos
	* code [here](https://github.com/taasinsaquib/fetch_urp)
* Goal was to create a dog model that could see a bone being thrown and then go find it
	* Used Unity
	* I worked on the retina model, which casts rays into the scene in a log-polar distribution
	* It takes in color from the objects hit by the rays, which is used to form the optic nerve vector, or ONV

# June - August 2021
* [code](retina)
* I didn't have access to a Windows machine at the time to set up/run the biomimetic eye model
	* was told that it was outdated, hard to use, etc.
	* so I tried to make my own model with a modern graphics library
* Open3d was the library of choice
* I created a pinhole approximation of an eye
	* the scene simply has a white ball on a black background, and it moves around
	* rays are cast through the pinhole into the scene to get an ONV as input
	* trained neural networks in Google Colab to output the angles that the ball moved
