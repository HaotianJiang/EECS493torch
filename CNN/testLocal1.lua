require "torch"
require "nn"
require "math"

-- global variables
learningRate = 0.01
learningRateDecay=0
maxIterations = 15
fraction = 0.01

batchsize=100
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

function downloadWeight(moduleIndex, network, network_server, downloadIndex)

    local mServer = network_server.modules[moduleIndex].weight:storage()
    local mNetwork = network.modules[moduleIndex].weight:storage()

    for i=1,mServer:size() do
      if downloadIndex[i] > 0 then
        mNetwork[i] = mServer[downloadIndex[i]]
      end
    end

    for i=1,mServer:size() do 
      downloadIndex[i]=0
    end

end



function uploadWeight(moduleIndex, accIndex, grads_acc, threshold, row, column, filename)

    local mGrad = grads_acc[accIndex]:storage()
    --local uploadGrad = torch.Tensor(mGrad:size())
    file = io.open(filename, "w")
    for i=1,mGrad:size() do 
      if math.abs(mGrad[i]) >=threshold then
        
        mGrad[i] = mGrad[i]
   
      else
        mGrad[i] = 0

      end
    end
    for t=1, row do
      for r=1, column do
        if r==column then
          file:write(grads_acc[accIndex][t][r], " ", "\n")
        else
          file:write(grads_acc[accIndex][t][r], " ")
        end
      end
    end
end

function saveWeight(moduleIndex, network, row, column, filename)

    file = io.open(filename, "w")

    for t=1, row do
      for r=1, column do
        if r==column then
          file:write(network.modules[moduleIndex].weight[t][r], " ", "\n")
        else
          file:write(network.modules[moduleIndex].weight[t][r], " ")
        end
      end
    end

end
function saveBias(moduleIndex, network, row, column, filename)
    print()
    file = io.open(filename, "w")

    for t=1, row do
      for r=1, column do
        if r==column then
          file:write(network.modules[moduleIndex].bias[t], " ", "\n")
        else
          file:write(network.modules[moduleIndex].bias[t], " ")
        end
      end
    end

end

function uploadBias(moduleIndex, accIndex, downloadIndex, grads_acc, threshold)

    local mGrad = grads_acc[accIndex]:storage()
    local uploadGrad = torch.Tensor(mGrad:size())

    for i=1,mGrad:size() do 
      if math.abs(mGrad[i]) >=threshold then
        
        downloadIndex[i]=i
        uploadGrad[i] = mGrad[i]
        --print(downloadIndex[i])
      else
        
        uploadGrad[i] = 0
        --print(downloadIndex[i])
      end
    end
    print(uploadGrad)
    print(downloadIndex)
end


function thresCal(grads_acc, percentage)

  local temp7 = grads_acc[7]:storage()
  local temp5 = grads_acc[5]:storage()
  local temp3 = grads_acc[3]:storage()
  local temp1 = grads_acc[1]:storage()
  local temp8 = grads_acc[8]:storage()
  local temp6 = grads_acc[6]:storage()
  local temp4 = grads_acc[4]:storage()
  local temp2 = grads_acc[2]:storage()
  size7=temp7:size()
  size5=temp5:size()
  size3=temp3:size()
  size1=temp1:size()
  size8=temp8:size()
  size6=temp6:size()
  size4=temp4:size()
  size2=temp2:size()
  local gradArray = torch.Tensor(size1+size3+size5+size7+size2+size4+size6+size8)

  for i=1, size7 do
      gradArray[i]=math.abs(temp7[i])
  end
  for i=size7+1, size7+size5 do
      gradArray[i]=math.abs(temp5[i-size7])
  end
  for i=size7+size5+1, size7+size5+size3 do
      gradArray[i]=math.abs(temp3[i-(size7+size5)])
  end
  for i=size7+size5+size3+1, size7+size5+size3+size1 do
      gradArray[i]=math.abs(temp1[i-(size7+size5+size3)])
  end
  for i=size7+size5+size3+size1+1, size7+size5+size3+size1+size2 do
      gradArray[i]=math.abs(temp7[i-(size7+size5+size3+size1)])
  end
  for i=size7+size5+size3+size1+size2+1, size7+size5+size3+size1+size2+size4 do
      gradArray[i]=math.abs(temp5[i-(size7+size5+size3+size1+size2)])
  end
  for i=size7+size5+size3+size1+size2+size4+1, size7+size5+size3+size1+size2+size4+size6 do
      gradArray[i]=math.abs(temp3[i-(size7+size5+size3+size1+size2+size4)])
  end
  for i=size7+size5+size3+size1+size2+size4+size6+1, size7+size5+size3+size1+size2+size4+size6+size8 do
      gradArray[i]=math.abs(temp1[i-(size7+size5+size3+size1+size2+size4+size6)])
  end
  local top, index = torch.topk(gradArray, math.ceil((size1+size3+size5+size7+size2+size4+size6+size8)*percentage), 1, true)
  local threshold = torch.min(top)

  return threshold
end


function test_predictor(predictor, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        
        --print( "----------------------" )
        --print( "Index Label Prediction" )
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
        local accuracy = 1-test_err
        --print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")
        print(accuracy)
end


-- main routine
function main()

        local dataIndex = arg[1]
        print("iteration:", dataIndex)

        weight1, weight4, weight8, weight10, weightbias1,weightbias4,weightbias8,weightbias10= dofile('read_weight.lua')

        training_dataset, testing_dataset, classes, classes_names = dofile('read_dataset.lua')
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

        ----seperate whole dataset into three parts---
        third= math.floor(training_dataset:size()/3)
        training_dataset1=subset(training_data,1,third)

        ----------------------------------------------
        --local network_1 = create_network(#classes)
        local network_1=torch.load("network.net")
        --renew bias
        for t=1, 10 do
          network_1.modules[1].bias[t]=weightbias1[t][1]
        end

        for t=1, 5 do
          network_1.modules[4].bias[t]=weightbias4[t][1]
        end

        for t=1, 300 do
          network_1.modules[8].bias[t]=weightbias8[t][1]
        end

        for t=1, 12 do
          network_1.modules[10].bias[t]=weightbias10[t][1]
        end

        for t=1, 10 do
          for r=1, 32 do
            network_1.modules[1].weight[t][r]=weight1[t][r]
          end
        end
        --renew weight
        for t=1, 5 do
          for r=1, 320 do
            network_1.modules[4].weight[t][r]=weight4[t][r]
          end
        end

        for t=1, 300 do
          for r=1, 585 do
            network_1.modules[8].weight[t][r]=weight8[t][r]
          end
        end

        for t=1, 12 do
          for r=1, 300 do
            network_1.modules[10].weight[t][r]=weight10[t][r]
          end
        end
      --print(dataIndex,': w1 ',network_1.modules[1].bias[1])
      local grads_acc1={torch.Tensor(10,32):zero(),torch.Tensor(10,1):zero(),torch.Tensor(5,320):zero(),torch.Tensor(5,1):zero(), torch.Tensor(300,585):zero(),torch.Tensor(300,1):zero(),torch.Tensor(12,300):zero(),torch.Tensor(12,1):zero()}
         
      print("---start training---")

        local startPoint = 1+dataIndex*batchsize
        local endPoint = dataIndex*batchsize+batchsize
        if endPoint<training_dataset1:size() then

          current_batch = subset(training_dataset1, startPoint, endPoint)

          ----------------------------------------------
          grads_acc1=table_add(grads_acc1,update(network_1, current_batch))           
        else
          print('I am done.')
          file = io.open('end.txt', "w")
          file:write('end')
        end        
        --print('gradients:',grads_acc1[2][1])
        --print(dataIndex,': w1 ',network_1.modules[1].bias[1])
        network_1.modules[1].gradBias=torch.round( network_1.modules[1].gradBias*100000000)*0.00000001
        network_1.modules[4].gradBias=torch.round( network_1.modules[4].gradBias*100000000)*0.00000001
        network_1.modules[8].gradBias=torch.round( network_1.modules[8].gradBias*100000000)*0.00000001
        network_1.modules[10].gradBias=torch.round( network_1.modules[10].gradBias*100000000)*0.00000001
        network_1.modules[1].gradWeight=torch.round( network_1.modules[1].gradWeight*100000000)*0.00000001
        network_1.modules[4].gradWeight=torch.round( network_1.modules[4].gradWeight*100000000)*0.00000001
        network_1.modules[8].gradWeight=torch.round( network_1.modules[8].gradWeight*100000000)*0.00000001
        network_1.modules[10].gradWeight=torch.round( network_1.modules[10].gradWeight*100000000)*0.00000001
        test_predictor(network_1, testing_dataset, classes, classes_names)

        local mthreshold = thresCal(grads_acc1, fraction)
        uploadWeight(10, 7, grads_acc1, mthreshold, 12, 300, "g4.txt")
        uploadWeight(8, 5, grads_acc1, mthreshold, 300, 585, "g3.txt")
        uploadWeight(4, 3, grads_acc1, mthreshold, 5, 320, "g2.txt")
        uploadWeight(1, 1, grads_acc1, mthreshold, 10, 32, "g1.txt")
        
        uploadWeight(10, 8, grads_acc1, mthreshold, 12, 1, "g8.txt")
        uploadWeight(8, 6, grads_acc1, mthreshold, 300, 1, "g7.txt")
        uploadWeight(4, 4, grads_acc1, mthreshold, 5, 1, "g6.txt")
        uploadWeight(1, 2, grads_acc1, mthreshold, 10, 1, "g5.txt")
        grads_acc1=nil    
        --grads_acc1={torch.Tensor(10,32):zero(),torch.Tensor(10):zero(),torch.Tensor(5,320):zero(),torch.Tensor(5):zero(), torch.Tensor(300,585):zero(),torch.Tensor(300):zero(),torch.Tensor(12,300):zero(),torch.Tensor(12):zero()}
        --[[saveWeight(10, network_1, 12, 300, "w4.txt")
        saveWeight(8, network_1, 300, 585, "w3.txt")
        saveWeight(4, network_1, 5, 320, "w2.txt")
        saveWeight(1, network_1, 10, 32, "w1.txt")
        saveBias(10, network_1, 12,1, "w8.txt")
        saveBias(8, network_1, 300, 1,"w7.txt")
        saveBias(4, network_1, 5, 1,"w6.txt")
        saveBias(1, network_1, 10,1, "w5.txt")]]
        
end


main()








