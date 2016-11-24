function Modelo = entrenarRNAClassication(X,Y,NumeroNeuronas)

%%% Completar el codigo %%%

hiddenLayerSize = NumeroNeuronas;
net = patternnet(hiddenLayerSize);
Modelo = train(net,X',Y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
