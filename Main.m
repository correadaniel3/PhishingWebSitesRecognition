
clc
clear all
close all

tic

load('X.mat');
load('XRF750.mat');
load('Y.mat');
load('XRNA.mat');
load('datosPhishing.mat');
load('resultadosSFS/RF750/featuresForRF.mat');
x=X;
%[residuals,reconstructed] = pcares(x,21);
%x=reconstructed(:,1:21);   Extraccion de variables
%x=XRF;
y=Y;

NumMuestras=size(x,1);
Rept=10;
punto=input('Ingrese 1 para k-vecinos, 2 para Random Forest, 3 para SVM, 4 para Gaussianas, 5 para redes neuronales, 6 para fisher, 7 para Pearson, 8 para SFS: ');
if punto==1
    sensibilidad=zeros(7,Rept);
    especificidad=zeros(7,Rept);
    precision=zeros(7,Rept);
    eficiencia=zeros(7,Rept);
    vecino=[1,3,5,7,9,11,13];
    tic
    for vecinos=1:7
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=x(particion.training(fold),:);
            Xtest=x(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));

            %%% Se normalizan los datos %%%
            [Xtrain,mu,sigma] = zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);

            Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,vecino(vecinos),'class');
            FN=(sum(Yesti~=Ytest))-(sum(Yesti==-1 & Yesti~=Ytest));
            FP=(sum(Yesti~=Ytest))-(sum(Yesti==1 & Yesti~=Ytest));
            TP=sum(Yesti==Ytest & Yesti==-1);
            TN=sum(Yesti==Ytest)-TP;
            sensibilidad(vecinos,fold)=(TP)/(TP+FN);
            especificidad(vecinos,fold)=(TN)/(TN+FP);
            precision(vecinos,fold)=(TP)/(TP+FP);
            eficiencia(vecinos,fold)=(TP+TN)/(TP+TN+FP+FN);
            texto=['vecinos = ', num2str(vecino(vecinos)),' fold: ',num2str(fold)];
            disp(texto);

        end
    end
    eficienciaFinalkn=zeros(7,2);
    especificidadFinalkn=zeros(7,2);
    sensibilidadFinalkn=zeros(7,2);
    precisionFinalkn=zeros(7,2);
    for i=1:7
        eficienciaFinalkn(i,1)=mean(eficiencia(i,:));
        eficienciaFinalkn(i,2)=std(eficiencia(i,:));
        especificidadFinalkn(i,1)=mean(especificidad(i,:));
        especificidadFinalkn(i,2)=std(especificidad(i,:));
        sensibilidadFinalkn(i,1)=mean(sensibilidad(i,:));
        sensibilidadFinalkn(i,2)=std(sensibilidad(i,:));
        precisionFinalkn(i,1)=mean(precision(i,:));
        precisionFinalkn(i,2)=std(precision(i,:));
    end
    toc
    save('resultadosKN/resultadosSeleccion2/eficienciaFinalkn.mat','eficienciaFinalkn');
    save('resultadosKN/resultadosSeleccion2/especificidadFinalkn.mat','especificidadFinalkn');
    save('resultadosKN/resultadosSeleccion2/sensibilidadFinalkn.mat','sensibilidadFinalkn');
    save('resultadosKN/resultadosSeleccion2/precisionFinalkn.mat','precisionFinalkn');

elseif punto==2
    sensibilidad=zeros(6,Rept);
    especificidad=zeros(6,Rept);
    precision=zeros(6,Rept);
    eficiencia=zeros(6,Rept);
    arbol=[50 100 250 500 750 1000];
    tic
    for arboles=1:6
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=x(particion.training(fold),:);
            Xtest=x(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));



            %%% Se normalizan los datos %%%
            [Xtrain,mu,sigma] = zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);
            
            modeloRF=entrenarFOREST(arbol(arboles),Xtrain,Ytrain');
            Yesti=testFOREST(modeloRF,Xtest);
            FN=(sum(Yesti~=Ytest))-(sum(Yesti==-1 & Yesti~=Ytest));
            FP=(sum(Yesti~=Ytest))-(sum(Yesti==1 & Yesti~=Ytest));
            TP=sum(Yesti==Ytest & Yesti==-1);
            TN=sum(Yesti==Ytest)-TP;
            sensibilidad(arboles,fold)=(TP)/(TP+FN);
            especificidad(arboles,fold)=(TN)/(TN+FP);
            precision(arboles,fold)=(TP)/(TP+FP);
            eficiencia(arboles,fold)=(TP+TN)/(TP+TN+FP+FN);
            texto=['Arboles = ', num2str(arbol(arboles)),' fold: ',num2str(fold)];
            disp(texto);

        end
    end
    eficienciaFinalrf=zeros(6,2);
    especificidadFinalrf=zeros(6,2);
    sensibilidadFinalrf=zeros(6,2);
    precisionFinalrf=zeros(6,2);
    for i=1:6
        eficienciaFinalrf(i,1)=mean(eficiencia(i,:));
        eficienciaFinalrf(i,2)=std(eficiencia(i,:));
        especificidadFinalrf(i,1)=mean(especificidad(i,:));
        especificidadFinalrf(i,2)=std(especificidad(i,:));
        sensibilidadFinalrf(i,1)=mean(sensibilidad(i,:));
        sensibilidadFinalrf(i,2)=std(sensibilidad(i,:));
        precisionFinalrf(i,1)=mean(precision(i,:));
        precisionFinalrf(i,2)=std(precision(i,:));
    end
    toc
    save('resultadosRF/resultadosSeleccion/eficienciaFinalrf.mat','eficienciaFinalrf');
    save('resultadosRF/resultadosSeleccion/especificidadFinalrf.mat','especificidadFinalrf');
    save('resultadosRF/resultadosSeleccion/sensibilidadFinalrf.mat','sensibilidadFinalrf');
    save('resultadosRF/resultadosSeleccion/precisionFinalrf.mat','precisionFinalrf');
    
elseif punto==3
    sensibilidad=zeros(5,Rept);
    especificidad=zeros(5,Rept);
    precision=zeros(5,Rept);
    eficiencia=zeros(5,Rept);
    gamma=[0.01 0.1 1 10 100];
    box=[0.01 0.1 1 10 100];
    tic
    for boxind=1:5
        for gammaind=1:5
            for fold=1:Rept
                rng('default');
                particion=cvpartition(NumMuestras,'Kfold',Rept);
                indices=particion.training(fold);
                Xtrain=x(particion.training(fold),:);
                Xtest=x(particion.test(fold),:);
                Ytrain=y(particion.training(fold));
                Ytest=y(particion.test(fold));

                %%% Se normalizan los datos %%%
                [Xtrain,mu,sigma] = zscore(Xtrain);
                Xtest=normalizar(Xtest,mu,sigma);
                
                Ytrain1=Ytrain;
                Ytrain1(Ytrain~=1)=-1;
                modelo1=entrenarSVM(Xtrain,Ytrain1,'classification',box(boxind),gamma(gammaind));
                Ytrain2=Ytrain;
                Ytrain2(Ytrain~=-1)=-1;
                Ytrain2(Ytrain==-1)=1;
                modelo2=entrenarSVM(Xtrain,Ytrain2,'classification',box(boxind),gamma(gammaind));
                [~,Yest1]=testSVM(modelo1,Xtest);
                [~,Yest2]=testSVM(modelo2,Xtest);            
                [~,Yesti] =max([Yest1,Yest2],[],2); 
                MatrizConfusion=zeros(2,2);
                for i=1:size(Xtest,1)
                    posTest= 1;
                    if Ytest(i)== -1
                        posTest = 2;
                    end
                    MatrizConfusion(Yesti(i),posTest) = MatrizConfusion(Yesti(i),posTest) + 1;
                end
                TP=MatrizConfusion(2,2);
                TN=MatrizConfusion(1,1);
                FN=MatrizConfusion(1,2);
                FP=MatrizConfusion(2,1);
                sensibilidad(gammaind,fold)=(TP)/(TP+FN);
                especificidad(gammaind,fold)=(TN)/(TN+FP);
                precision(gammaind,fold)=(TP)/(TP+FP);
                eficiencia(gammaind,fold)=(TP+TN)/(TP+TN+FP+FN);

                texto=['Gamma = ', num2str(gamma(gammaind)),' fold: ',num2str(fold), ' Box: ',num2str(box(boxind))];
                disp(texto);

            end
        end
        eficienciaFinalsvm=zeros(5,2);
        especificidadFinalsvm=zeros(5,2);
        sensibilidadFinalsvm=zeros(5,2);
        precisionFinalsvm=zeros(5,2);
        for i=1:5
            eficienciaFinalsvm(i,1)=mean(eficiencia(i,:));
            eficienciaFinalsvm(i,2)=std(eficiencia(i,:));
            especificidadFinalsvm(i,1)=mean(especificidad(i,:));
            especificidadFinalsvm(i,2)=std(especificidad(i,:));
            sensibilidadFinalsvm(i,1)=mean(sensibilidad(i,:));
            sensibilidadFinalsvm(i,2)=std(sensibilidad(i,:));
            precisionFinalsvm(i,1)=mean(precision(i,:));
            precisionFinalsvm(i,2)=std(precision(i,:));
        end
        toc
        texto1=['resultadosSVM/resultadosSeleccion/eficienciaFinalsvm',num2str(boxind),'.mat'];
        texto2=['resultadosSVM/resultadosSeleccion/especificidadFinalsvm',num2str(boxind),'.mat'];
        texto3=['resultadosSVM/resultadosSeleccion/sensibilidadFinalsvm',num2str(boxind),'.mat'];
        texto4=['resultadosSVM/resultadosSeleccion/precisionFinalsvm',num2str(boxind),'.mat'];
        save(texto1,'eficienciaFinalsvm');
        save(texto2,'especificidadFinalsvm');
        save(texto3,'sensibilidadFinalsvm');
        save(texto4,'precisionFinalsvm');
    end
    
elseif punto==4
    eficiencia=zeros(1,Rept); %%vector fila con "rept" elementos
    sensibilidad=zeros(1,Rept);
    especificidad=zeros(1,Rept);
    precision=zeros(1,Rept);

    NumClases=2;
    Mezclas=1;
    for fold=1:Rept

        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept); %%Validacion cruzada, k sera Rept, K subconjuntos de "igual" tamaño
        indices=particion.training(fold);  %%Retorna un vector logico que indica que muestras son para entrenar (training) y cuales para validar (test)

        Xtrain=x(particion.training(fold),:);
        Xtest=x(particion.test(fold),:);
        Ytrain=y(particion.training(fold));
        Ytest=y(particion.test(fold));
        
        [Xtrain,mu,sigma] = zscore(Xtrain);
        %%testing=repmat(mu,size(Xtest,1),1);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1); %%(Xtest - media)/std

        vInd=(Ytrain == 1);  %%Obtiene los indices en la salida Y que corresponden a la clase 1
        XtrainC1 = Xtrain(vInd,:); %%Obtiene las muestras de entrenamiento de la clase 1
        if ~isempty(XtrainC1)
            Modelo1=entrenarGMM(XtrainC1,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end

        vInd=(Ytrain == -1); %%Obtiene los indices en la salida Y que corresponden a la clase 2
        XtrainC2 = Xtrain(vInd,:); %%Obtiene las muestras de entrenamiento de la clase 2
        if ~isempty(XtrainC2)
            Modelo2=entrenarGMM(XtrainC2,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end        

        probClase1=testGMM(Modelo1,Xtest);
        probClase2=testGMM(Modelo2,Xtest);        
        Matriz=[probClase1,probClase2]; %%Matriz con 2 columnas, cada una es la probabilidad en cada clase de las muestras de validacion

        [~,Yest] = max(Matriz,[],2); 
        MatrizConfusion = zeros(NumClases,NumClases);

        for i=1:size(Xtest,1)
            posTest= 1;
            if Ytest(i)== -1
                posTest = 2;
            end
            MatrizConfusion(Yest(i),posTest) = MatrizConfusion(Yest(i),posTest) + 1;
        end
        TP=MatrizConfusion(2,2);
        TN=MatrizConfusion(1,1);
        FN=MatrizConfusion(1,2);
        FP=MatrizConfusion(2,1);
        sensibilidad(1,fold)=(TP)/(TP+FN);
        especificidad(1,fold)=(TN)/(TN+FP);
        precision(1,fold)=(TP)/(TP+FP);
        eficiencia(1,fold)=(TP+TN)/(TP+TN+FP+FN);

    end
    eficienciaFinalGauss=zeros(1,2);
    especificidadFinalGauss=zeros(1,2);
    sensibilidadFinalGauss=zeros(1,2);
    precisionFinalGauss=zeros(1,2);
    
    eficienciaFinalGauss(1,1)=mean(eficiencia(1,:));
    eficienciaFinalGauss(1,2)=std(eficiencia(1,:));
    especificidadFinalGauss(1,1)=mean(especificidad(1,:));
    especificidadFinalGauss(1,2)=std(especificidad(1,:));
    sensibilidadFinalGauss(1,1)=mean(sensibilidad(1,:));
    sensibilidadFinalGauss(1,2)=std(sensibilidad(1,:));
    precisionFinalGauss(1,1)=mean(precision(1,:));
    precisionFinalGauss(1,2)=std(precision(1,:));
    toc
    save('resultadosGauss/eficienciaFinalGauss.mat','eficienciaFinalGauss');
    save('resultadosGauss/especificidadFinalGauss.mat','especificidadFinalGauss');
    save('resultadosGauss/sensibilidadFinalGauss.mat','sensibilidadFinalGauss');
    save('resultadosGauss/precisionFinalGauss.mat','precisionFinalGauss');
elseif punto==5
    eficiencia=zeros(1,Rept); %%vector fila con "rept" elementos
    sensibilidad=zeros(1,Rept);
    especificidad=zeros(1,Rept);
    precision=zeros(1,Rept);
    NumNeuronas=[10,20,30,40,50,100,150];
    redesY = zeros(NumMuestras,2);
    tic
    for n=1:7
        for i=1:NumMuestras
            if(y(i)==1)
                redesY(i,1)=1;
            else
                redesY(i,2)=1;
            end
        end
        for fold=1:Rept

            %%% Se hace la partición de las muestras %%%
            %%%      de entrenamiento y prueba       %%%

            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=x(particion.training(fold),:);
            Xtest=x(particion.test(fold),:);
            Ytrain = redesY(particion.training(fold),:);
            [~,Ytest]=max(redesY(particion.test(fold),:),[],2);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Se normalizan los datos %%%

            [XtrainNormal,mu,sigma]=zscore(Xtrain);
            XtestNormal=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

            Modelo=entrenarRNAClassification(Xtrain,Ytrain,NumNeuronas(n));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Validación de los modelos. %%%

            Yesti=testRNA(Modelo,Xtest);
            [~,Yesti]=max(Yesti,[],2);

            FN=sum(Yesti==2 & Yesti~=Ytest);
            FP=sum(Yesti==1 & Yesti~=Ytest);
            TP=sum(Yesti==Ytest & Yesti==1);
            TN=sum(Yesti==Ytest)-TP;
            sensibilidad(n,fold)=(TP)/(TP+FN);
            especificidad(n,fold)=(TN)/(TN+FP);
            precision(n,fold)=(TP)/(TP+FP);
            eficiencia(n,fold)=(TP+TN)/(TP+TN+FP+FN); 
            texto=['Neuronas = ', num2str(NumNeuronas(n)),' fold: ',num2str(fold)];
            disp(texto);
        end   
    end
    eficienciaFinalRNA=zeros(7,2);
    especificidadFinalRNA=zeros(7,2);
    sensibilidadFinalRNA=zeros(7,2);
    precisionFinalRNA=zeros(7,2);
    for i=1:7
        eficienciaFinalRNA(i,1)=mean(eficiencia(i,:));
        eficienciaFinalRNA(i,2)=std(eficiencia(i,:));
        especificidadFinalRNA(i,1)=mean(especificidad(i,:));
        especificidadFinalRNA(i,2)=std(especificidad(i,:));
        sensibilidadFinalRNA(i,1)=mean(sensibilidad(i,:));
        sensibilidadFinalRNA(i,2)=std(sensibilidad(i,:));
        precisionFinalRNA(i,1)=mean(precision(i,:));
        precisionFinalRNA(i,2)=std(precision(i,:));
    end
    toc
    save('resultadosRN/eficienciaFinalRNA.mat','eficienciaFinalRNA');
    save('resultadosRN/especificidadFinalRNA.mat','especificidadFinalRNA');
    save('resultadosRN/sensibilidadFinalRNA.mat','sensibilidadFinalRNA');
    save('resultadosRN/precisionFinalRNA.mat','precisionFinalRNA');
    
elseif punto==6
    dataForClasses = classificatedata(X, Y);
    fishers = calculatefisher(dataForClasses);
    posibleEvaluar = fishers(fishers<0.004);
    save('resultadosFisher/posibleEvaluar.mat','posibleEvaluar');
    
elseif punto==7
    temp = [X, Y];
    correlationMatrixWithOutput = corrcoef(temp);
    save('resultadosPearson/CorrelationMatrixWithOutput.mat', 'correlationMatrixWithOutput');
elseif punto==8
    tic
    y(y==-1)=0;
    cvp = cvpartition(Y(:, 1), 'k', 10);
    featuresForRF = sequentialfs(@functionRF, x, y(:, 1), 'cv', cvp);
    toc
    save('resultadosSFS/featuresForRF.mat', 'featuresForRF');

end

