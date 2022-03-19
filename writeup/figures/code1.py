def forward(self, x):
	x = self.fc1(x)
	x = torch.relu(x)

	x = self.fc2(x)
	x = torch.relu(x)

	# Output layer
	out = self.fc3(x)
	return out
	
def forward(self, xTensor):
	# Init LIF Layers to replace Relu
	mem0 = self.lif1.init_leaky()
	mem1 = self.lif2.init_leaky()

	out = None

	for t in range(numSteps):
		x = xTensor[t]

		x = self.fc1(x)
		x, mem1 = self.lif1(x, mem1)

		x = self.fc2(x)
		x, mem2 = self.lif2(x, mem2)

		# Output layer
		out = self.fc3(mem2)
	return out

def forward(self, xTensor)

	# Init LIF Layers to replace Relu
	mem0 = self.lif0.init_leaky()
	mem1 = None		# Hold result of SNN

	# SNN Part **********************************

	for t in range(numSteps):
		x = xTensor[t]

		x = self.fc1(x)
		x, mem1 = self.lif0(x, mem1)

	# ANN Part **********************************

	x = self.fc2(mem1)
	x = torch.relu(x)

	# Output layer
	out = self.fc3(x)

	return out