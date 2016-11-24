function error = functionKN(Xtrain, Ytrain, Xtest, Ytest)
    [Xtrain, means, stds] = zscore(Xtrain);
    Xtest = normalize(Xtest, means, stds);
    k = 3;
    Yest = zeros(size(Xtest, 1), 1);

    for i = 1:size(Xtest, 1)
        distances = calculateeuclideandistance(Xtrain, Xtest(i, :));
        [distances, sortIndexes] = sort(distances);
        temp = Ytrain(sortIndexes(1 : k));

        Yest(i) = mean(temp);
    end
    error = calculateerrorKN(Ytest, Yest, 1);