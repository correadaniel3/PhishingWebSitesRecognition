function error = functionRF(Xtrain, Ytrain, Xtest, Ytest)



%%% punto Random Forest %%%
	
    NumClases=length(unique(Ytrain)); %%% Se determina el número de clases del problema.
    NumArboles=750;

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validación de los modelos. %%%
        
        Yest=testFOREST(Modelo,Xtest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion = zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i)+1,Ytest(i)+1) = MatrizConfusion(Yest(i)+1,Ytest(i)+1) + 1;
        end
        Eficiencia = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    
   %% Eficiencia=(sum(Yesti==Ytest))/length(Ytest);



   
    
    error=1-Eficiencia;
    disp(num2str(error));
	%%% Fin punto Random Forest %%%