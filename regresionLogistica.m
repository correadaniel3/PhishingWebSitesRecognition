function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);

for iter = 1:1000
    %%% Completar el código %%% 
    W = W - ((eta/N)*(sigmoide((W'*X')') - Y)'*X)';
    %%% Fin de la modificación %%%
end

end