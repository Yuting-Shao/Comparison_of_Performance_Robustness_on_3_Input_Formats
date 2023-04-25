- Yuting Shao, shao.yut@northeastern.edu

- The OS that I use: macOS Ventura 13.2

- The IDE that I use: Sublime Text

- Run command: 
	For performace comparison and hyperparameters optimization of ResNet-34: python resnet34.py

	For performace comparison of VGG16: python vgg16.py

	For variations in intensity: python intensity.py

	For variations in color balance: python colorbalance.py

	For variations in orientations: python orientation.py

	For all variations: python robustness.py


- Note:
	The main code can be found at resnet34.py. vgg16.py is similar to resent34.py, but uses the VGG16 model instead of ResNet-34.

	intensity.py, colorbalance.py, orientation.py, and robustness.py are similar, but apply different variations to the testing image.

	Before running intensity.py, colorbalance.py, orientation.py, and robustness.py, make sure that the trained model has been saved for use.
