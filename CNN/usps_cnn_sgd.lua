
require "torch"
require "nn"
require "math"

-- global variables
learningRate = 0.01
maxIterations = 7700


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
                      print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        local test_err = mistakes/tested_samples
        print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


-- main routine
function main()

        local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
 
        local network = create_network(#classes)
		--print(training_dataset[1])
	 --  print(training_dataset[1])
		--print(torch.Tensor(64,1))
	  -- print(network:forward(training_dataset[1][1]))
	--	print(network:forward(torch.Tensor(64,1)))
		--print(network)
       --print (table.getn(testing_dataset))
      train_network(network, training_dataset)
        
      test_predictor(network, testing_dataset, classes, classes_names)

end


main()
















