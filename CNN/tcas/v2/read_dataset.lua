
require "torch"
require "image"
require "math"

classes = {1, 2}
classes_names = {'0','1'}


function loadtxt(datafile, labelfile)
	local datafile,dataerr=io.open(datafile,"r")
	local labelfile,labelerr=io.open(labelfile,"r")
	if dataerr then print("Open data file Error")	end
	if labelerr then print("Open label file Error")	end
	local index = 0
    local dataset={}
	while true do
		local dataline=datafile:read('*l')
		local labelline=labelfile:read('*l')
		if dataline == nil then break end
		index=index+1
		--print (index)
		data1=dataline:split(' ')
		data=torch.Tensor(135,1)
		--data={}
		for i=1,135 do
			data[i]=tonumber(data1[i])
		end
		label1=labelline:split(' ')
		label=torch.Tensor(1)
		--label={}
		for i=1,table.getn(label1) do
			label[i]=tonumber(label1[i])
		end     
		dataset[index]={data,label}
	end
	function dataset:size() return index end
	return dataset
end

local training_dataset = loadtxt('tcas_parameter.txt','tcas_label2.txt')
local testing_dataset  = loadtxt('tcas_testingSet.txt','tcas_testingLabel.txt')
--a={{1,2,2},{3,4,5}}
--print("one data")    
--print(type(testing_dataset[1][2]))    
--print(testing_dataset[1][2])
--print (table.getn(testing_dataset))


return training_dataset, testing_dataset, classes, classes_names















