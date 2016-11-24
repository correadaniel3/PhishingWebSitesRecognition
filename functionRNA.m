function error = functionRNA(Xtrain, Ytrain, Xtest, Ytest)
    [Xtrain, means, stds] = zscore(Xtrain);
    Xtest = normalize(Xtest, means, stds);

    net = newff(Xtrain', Ytrain', 10, {'tansig', 'purelin'}, 'trainlm');
    net.trainParam.epochs = 50;

    [net, TR] = train(net, Xtrain', Ytrain');
    Yest = sim(net, Xtest');

    error = calculateerrorRNA(Ytest, Yest');