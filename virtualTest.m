m=253;
Q = eye(m); 
fileID = fopen('totinfo_testingSet.txt','wt');
for i=1:1:m
    fprintf(fileID,'%d ',Q(i,:));
    fprintf(fileID,'\n');
end
fclose(fileID);

fileID = fopen('totinfo_testingLabel.txt','wt');
for i=1:1:m
    fprintf(fileID,'%d',0);
    fprintf(fileID,'\n');
end
fclose(fileID);


