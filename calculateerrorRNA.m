function error = calculateerrorRNA(Yval, Yest)
    % Función para calcular el error en dos vectores columnas.
    error = sum(sum((Yval - Yest) .^ 2, 2)) / 2;