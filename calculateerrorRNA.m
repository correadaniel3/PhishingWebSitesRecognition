function error = calculateerrorRNA(Yval, Yest)
    % Funci�n para calcular el error en dos vectores columnas.
    error = sum(sum((Yval - Yest) .^ 2, 2)) / 2;