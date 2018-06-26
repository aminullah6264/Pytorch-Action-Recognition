TotalFeaturesY=TrainLables;
Tlables=zeros(length(TotalFeaturesY),6);
for i=1:length(TotalFeaturesY)
    for j=1:6
    if TotalFeaturesY(i,1)==j
        Tlables(i,j)=1;
    end
    end
end



TotalFeaturesY=ValLables;
Vlables=zeros(length(TotalFeaturesY),6);
for i=1:length(TotalFeaturesY)
    for j=1:6
    if TotalFeaturesY(i,1)==j
        Vlables(i,j)=1;
    end
    end
end


