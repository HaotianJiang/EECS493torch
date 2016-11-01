
require "torch"
require "nn"
require "math"

-- global variables
learningRate = 0.01
--maxIterations = 7700
learningRateDecay=0
maxIterations = 15

-- here we set up the architecture of the neural network
function create_network()

   local ann = nn.Sequential()  -- make a multi-layer structure
                         
   ann:add(nn.TemporalConvolution(1,10,26))   -- 135*1 goes in, 110*10 goes out
   ann:add(nn.TemporalMaxPooling(2)) -- becomes  55*10 
   ann:add(nn.ReLU())  
   
   ann:add(nn.TemporalConvolution(10,5,25))  -- 55*10 goes in, 31*5 goesout
   ann:add(nn.TemporalMaxPooling(2)) -- becomes  15*5 
   ann:add(nn.ReLU())  

   ann:add(nn.View(15*5))
   ann:add(nn.Linear(15*5,300))
   ann:add(nn.ReLU())
   ann:add(nn.Linear(300,2))
   ann:add(nn.ReLU())
   ann:add(nn.LogSoftMax())
   
   return ann
end


-- train a Neural Netowrk
function train_network(network, dataset, testing_dataset)
   local criterion = nn.ClassNLLCriterion()
   iteration=1
   while true do
	   local currentError=0
	   for t=1, table.getn(dataset) do
           local example = dataset[t]
           local input = example[1]
           local target = example[2]
           currentError = currentError + criterion:forward(network:forward(input), target)
           network:updateGradInput(input, criterion:updateGradInput(network.output, target))
           network:zeroGradParameters()
           network:accGradParameters(input, criterion.gradInput, 1)
           network:updateParameters(learningRate)
	   end
     currentError = currentError / table.getn(dataset)
     iteration = iteration + 1
     currentLearningRate = learningRate/(1+iteration*learningRateDecay)
     if iteration > maxIterations then
        break
     end
     --test_predictor(network, testing_dataset, classes, classes_names)
   end
end

--[[function train_network(network, dataset)
               
   print( "Training the network" )
   local criterion = nn.ClassNLLCriterion()
   size=table.getn(dataset)

   trainer=nn.StochasticGradient(network,criterion)
   trainer.learningRate=0.01
   trainer.maxIteration=10
   trainer:train(dataset)

end--]]

function saveBias(moduleIndex, network, row, column, filename)
    print()
    file = io.open(filename, "w")
    file:write(i, " ", probability, " ", prediction[1], "\n")
end


function test_predictor(predictor, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        
        --print( "----------------------" )
        --print( "Index Label Prediction" )
	file = io.open("result.txt", "w")
        for i=1,table.getn(test_dataset) do

               local input  = test_dataset[i][1]
               local class_id = test_dataset[i][2]        
               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               --print(probabilites_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 
               print(i, " ", probabilites_per_class[1], " ", probabilites_per_class[2])
	       --print(i, " ", probability, " ", prediction)
  	       
  	       file:write(tostring(probabilites_per_class[1]),"\n")
               if prediction[1] ~= class_id[1] then
                      mistakes = mistakes + 1
                      local label = classes_names[ classes[class_id[1]] ]
                      local predicted_label = classes_names[ classes[prediction[1] ] ]
                      --print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end
        file:close()
       -- local test_err = mistakes/tested_samples
       -- local accuracy = 1-test_err
        --print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")
       -- print(accuracy)
end


-- main routine
function main()

        local training_dataset, testing_dataset, classes, classes_names = dofile('read_dataset.lua')
        local network = create_network()
        train_network(network, training_dataset, testing_dataset)
        test_predictor(network, testing_dataset, classes, classes_names)

end


main()
















