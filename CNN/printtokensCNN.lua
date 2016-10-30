
require "torch"
require "nn"
require "math"

-- global variables
learningRate = 0.01
learningRateDecay=0
maxIterations = 15
fraction = 0.01

batchsize=100


-- here we set up the architecture of the neural network
function create_network(nb_outputs)

   local ann = nn.Sequential();  -- make a multi-layer structure
   
   -- 16x16x1                        
   ann:add(nn.TemporalConvolution(1,10,32))   -- 561*1 goes in, 498*10 goes out
   ann:add(nn.TemporalMaxPooling(2)) -- becomes  249*10 
   ann:add(nn.ReLU())  
   
   ann:add(nn.TemporalConvolution(10,5,32))  -- 249*10 goes in, 218*10 goesout
   ann:add(nn.TemporalMaxPooling(2)) -- becomes  109*10 
   ann:add(nn.ReLU())  
   --ann:add(nn.Reshape(6*6*6))
   ann:add(nn.View(117*5))
   ann:add(nn.Linear(117*5,300))
   ann:add(nn.ReLU())
   ann:add(nn.Linear(300,12))
   ann:add(nn.ReLU())
   ann:add(nn.LogSoftMax())
   
   return ann
end

function subset(dataset, head, tail)
  local sub={}
  local index=0
  for i=head, tail do
      index=index+1
      sub[index]=dataset[i]
  end
  function sub:size() return index end
  return sub
end

function table_add(table1, table2)
  if table.getn(table1)~= table.getn(table2) then
    print('not same size table')
  end
  sum={}
  for i=1, table.getn(table1) do
    sum[i]=table1[i]+table2[i]
  end
  return sum
end

-- accumulate gradients
function update(network, dataset)
   local criterion = nn.ClassNLLCriterion()
   local grads_acc={torch.Tensor(10,32):zero(),torch.Tensor(10):zero(),torch.Tensor(5,320):zero(),torch.Tensor(5):zero(), torch.Tensor(300,585):zero(),torch.Tensor(300):zero(),torch.Tensor(12,300):zero(),torch.Tensor(12):zero()}
   for iteration=1,maxIterations do
     local currentError=0
     for t=1, table.getn(dataset) do
           --local example = dataset[shuffledIndices[t]]
           local example = dataset[t]
           local input = example[1]
           local target = example[2]
           currentError = currentError + criterion:forward(network:forward(input), target)
           network:updateGradInput(input, criterion:updateGradInput(network.output, target))
           network:zeroGradParameters()
           network:accGradParameters(input, criterion.gradInput, 1)
           local weights,grads=network:parameters()
          --print("gradients: ",grads)
           grads_acc=table_add(grads_acc,grads)
           network:updateParameters(learningRate)
     end
     --test_predictor(network, testing_dataset, classes, classes_names)
   end
   return grads_acc
end

-- train a Neural Netowrk
function train_network(network, dataset)
               
   print( "Training the network" )
   local criterion = nn.ClassNLLCriterion()
   size=table.getn(dataset)
  --[[ for times=1,30 do
   for iteration=1, table.getn(dataset)do
	  -- local index = math.random(size) -- pick example at random
	   local index=iteration
	   local index = iteration
       local input = dataset[index][1]               
       local output = dataset[index][2]
    --   print(input)
       criterion:forward(network:forward(input), output)
	   --print(output)
       network:zeroGradParameters()
       network:backward(input, criterion:backward(network.output, output))
       network:updateParameters(learningRate)
   end
   end--]]
   trainer=nn.StochasticGradient(network,criterion)
   trainer.learningRate=0.01
   trainer.maxIteration=10
   trainer:train(dataset)
   --network:forward(dataset[1][1]))
  -- print(dataset[1][2])--]]

end


function test_predictor(predictor, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        
        print( "----------------------" )
        print( "Index Label Prediction" )
        for i=1,table.getn(test_dataset) do

               local input  = test_dataset[i][1]
               local class_id = test_dataset[i][2]        
               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 
                      
               if prediction[1] ~= class_id[1] then
                      mistakes = mistakes + 1
                      local label = classes_names[ classes[class_id[1]] ]
                      local predicted_label = classes_names[ classes[prediction[1] ] ]
                      --print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        local test_err = mistakes/tested_samples
        print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


-- main routine
function main()
        --local dataIndex = arg[1]
        --print("iteration:", dataIndex)

        local training_dataset, testing_dataset, classes, classes_names = dofile('read_dataset.lua')
         ----shuffle the order of the whole dataset----

        training_data={}
        ----------------------------------------------
        local shuffledIndices={}
        local f = io.open("randomList.txt")
        for line in f:lines() do
          table.insert(shuffledIndices, tonumber(line))
        end

        for i=1,training_dataset:size() do
          local example = training_dataset[shuffledIndices[i]]
          training_data[i]=example  
        end

        local network = create_network(#classes)

        -- testing merge 1234567  12345  masterIn

        --local grads_acc={torch.Tensor(10,32):zero(),torch.Tensor(10,1):zero(),torch.Tensor(5,320):zero(),torch.Tensor(5,1):zero(), torch.Tensor(300,585):zero(),torch.Tensor(300,1):zero(),torch.Tensor(12,300):zero(),torch.Tensor(12,1):zero()}
         
        print("---start training---")
        for dataIndex=1,1000 do
		local startPoint = 1+dataIndex*batchsize
		local endPoint = dataIndex*batchsize+batchsize
		if endPoint<training_dataset:size() then

		  current_batch = subset(training_dataset, startPoint, endPoint)
                  train_network(network, current_batch)
		  ----------------------------------------------
		  --grads_acc=table_add(grads_acc,update(network, current_batch))           
		else
		  print('I am done.')
		  file = io.open('end.txt', "w")
		  file:write('end')
		  break
		end  
		test_predictor(network, testing_dataset, classes, classes_names)
        end
        

end


main()
















