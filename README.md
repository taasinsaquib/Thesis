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

# August - September 2021
* [code] 
* received code from Masaki for the locally connected networks used in Arjun's thesis 
* got that set up and tried to train it on data collected from my Open3d scene
* looked into SNN libraries
	* started with NengoDL, then found snnTorch

# September - October 2021
* [code] 
* got a Windows machine from the lab (yay!)
* was finally able to try compiling the siggraph eye demo
	* wasted about two weeks on this before I realized that I needed to download Visual Studio 10
		* you actually can't download it anymore, so maybe I'll host the downloader somewhere
	* I was re-compiling libraries and a lot of other stuff to deal with weird compiler errors
* once I got the eye working I got more familiar with the code and collecting training data

# October - December 2021
* [code](trainSNNs), [code](RunSNN)
* after getting another machine from the lab (this time with a GPU), I started training my SNNs
* I also had to find a way to connect my Python code to the VS10 project
	* libraries exist, but the eye code was just too old
	* I had to compile my code into an exe and then call that from the eye project
		* pyinstaller is a great library for this
