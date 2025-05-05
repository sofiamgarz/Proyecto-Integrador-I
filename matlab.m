% Descripción: Importa datos, imputa valores faltantes usando Python, normaliza,
% aplica PCA y clustering K-means en MATLAB.

clear; clc; close all; % Limpiar entorno

% PASOS PREVIOS
fprintf('PASOS PREVIOS\n');
% Importamos la base de datos inicial
opts = detectImportOptions('datasets/Monografia_final.csv');
% Especifica que las celdas vacías se traten como NaN para columnas numéricas
numericCols = {'Tasa_paro', 'Densidad_poblacion', 'N_empresas', ...
               'Transacciones_inmobiliarias', 'Deuda_ayuntamientos', ...
               'Saldo_migratorio', 'Renta_disponible_media'};
opts = setvartype(opts, numericCols, 'double');
opts = setvaropts(opts, numericCols, 'TreatAsMissing', ''); % Tratar strings vacías como missing

try
    df = readtable('datasets/Monografia_final.csv', opts);
    disp('Primeras 5 filas del DataFrame original:');
    disp(head(df, 5));
catch ME
    fprintf('Error cargando datasets/Monografia_final.csv.\n');
    fprintf('Error: %s\n', ME.message);
    return; % Detener si no se puede cargar
end


%% Paso 2: Imputación de datos faltantes con Python
disp('Ejecutando Paso 2: Imputación (el resultado NO se usará directamente en pasos siguientes)');
% Definir columnas a imputar
columnas_imputar = {'N_empresas','Renta_disponible_media','Tasa_paro', ...
                  'Densidad_poblacion','Transacciones_inmobiliarias', ...
                  'Deuda_ayuntamientos','Saldo_migratorio'};

% Verificar que las columnas a imputar existan en df
cols_exist_original = ismember(columnas_imputar, df.Properties.VariableNames);
if ~all(cols_exist_original)
    warning('Las siguientes columnas para imputar no existen en df: %s. Saltando imputación.', strjoin(columnas_imputar(~cols_exist_original), ', '));
else
    % Configurar entorno Python (necesario tener scikit-learn instalado)
    try
        pe = pyenv;
        fprintf('Usando entorno Python: %s\n', pe.Version);
        % Importar IterativeImputer directamente
        iter_mod = py.importlib.import_module('sklearn.impute');
        IterImp = iter_mod.IterativeImputer(pyargs('max_iter', int32(11), 'random_state', int32(0)));

        % Seleccionamos las columnas que van a participar en la imputación
        X_train_table = df(:, columnas_imputar);
        X_train_mat = table2array(X_train_table); % Convertir a matriz numérica

        % Convertir la matriz de MATLAB a un array de NumPy para sklearn
        X_train_py = py.numpy.array(X_train_mat);

        % Ajustamos y transformamos con los datos reales
        IterImp.fit(X_train_py);
        datos_imputados_py = IterImp.transform(X_train_py);

        % Convertimos el resultado de nuevo a una matriz de MATLAB
        datos_imputados_mat = double(datos_imputados_py);
        disp('Primeras 5 filas de datos imputados (matriz):');
        disp(head(datos_imputados_mat, 5));

        % Reemplazamos las columnas en la tabla df con los datos imputados
        % NOTA: Este df modificado NO se usa más adelante porque se carga
        % 'FINAL_DATOS_IMPUTADOS-2.csv' en el siguiente paso.
        for i = 1:length(columnas_imputar)
            df.(columnas_imputar{i}) = datos_imputados_mat(:, i);
        end
        disp('Tabla df actualizada con datos imputados (pero no se usará después).');

    catch ME
         warning('Error durante la imputación con Python en Paso 2: %s. Saltando este paso.');
         disp('Detalles del error de Python:');
         disp(ME.cause); % Puede dar más info si el error viene de Python
         if isa(ME.cause, 'py.Exception')
             disp(char(ME.cause.args));
         end
    end
end


% NORMALIZACIÓN ESTÁNDAR
fprintf('\n NORMALIZACIÓN ESTÁNDAR \n');
% Cargamos el dataset que ya tiene los datos imputados Y CORREGIDOS (N_empresas >= 0)
try
    Datos_imputados = readtable('datasets/FINAL_DATOS_IMPUTADOS-2.csv');
    disp('Primeras 5 filas del DataFrame imputado y corregido:');
    disp(head(Datos_imputados, 5));
catch ME
    fprintf('Error cargando datasets/FINAL_DATOS_IMPUTADOS-2.csv.\n');
    fprintf('Asegúrate de que el archivo exista y contenga los datos imputados con N_empresas >= 0.\n');
    fprintf('Error: %s\n', ME.message);
    return; % Detener si no se puede cargar el archivo necesario
end

% Columnas a no estandarizar
columnas_no_std_nombres = {'Municipio', 'Codigo_Postal'};
columnas_no_std = Datos_imputados(:, columnas_no_std_nombres);

% Seleccionar columnas a estandarizar (todas excepto las anteriores)
nombres_variables = Datos_imputados.Properties.VariableNames;
columnas_a_std_nombres = setdiff(nombres_variables, columnas_no_std_nombres, 'stable');
dataset_table = Datos_imputados(:, columnas_a_std_nombres);

% Convertir a matriz para estandarizar
dataset_mat = table2array(dataset_table);

% Verificar si hay NaNs o Infs antes de zscore
if any(isinf(dataset_mat(:))) || any(isnan(dataset_mat(:)))
    warning('Se encontraron valores NaN o Inf en los datos antes de la estandarización. Esto puede causar problemas.');
    % Opcional: Manejar NaNs/Infs aquí si es necesario, aunque se supone que vienen de un archivo ya imputado.
    % dataset_mat(isnan(dataset_mat)) = 0; % Ejemplo: reemplazar NaN con 0 (¡cuidado!)
end

% Aplicar estandarización (z-score en MATLAB)
standardized_dataset_mat = zscore(dataset_mat);

% Mostrar resultados de la estandarización
disp('Muestra de base de datos estandarizada (matriz):');
disp(head(standardized_dataset_mat, 5));
fprintf('Media aritmética (estandarizada): %.4f\n', mean(standardized_dataset_mat(:), 'omitnan')); 
fprintf('Varianza (estandarizada): %.4f\n', var(standardized_dataset_mat(:), 1, 'omitnan')); 
fprintf('Desviación estándar (estandarizada): %.4f\n', std(standardized_dataset_mat(:), 1, 'omitnan')); 

% Convertir la matriz estandarizada de nuevo a tabla
df_standardized = array2table(standardized_dataset_mat, 'VariableNames', columnas_a_std_nombres);

% Concatenar con las columnas no estandarizadas
df_final = [df_standardized, columnas_no_std];
% df_final = df_final(:, Datos_imputados.Properties.VariableNames);

disp('Primeras 5 filas de la tabla final estandarizada:');
disp(head(df_final,5));

% Guardar la tabla estandarizada
try
    writetable(df_final, 'datasets/standardized_dataset_matlab.csv');
    fprintf('Datos estandarizados guardados en datasets/standardized_dataset_matlab.csv\n');
catch ME
    fprintf('Error guardando standardized_dataset_matlab.csv: %s\n', ME.message);
    % Continuar si el guardado falla, pero advertir.
end

% PCA (Análisis de Componentes Principales)
fprintf('\n PCA \n');
% Usaremos df_final que acabamos de crear, en lugar de recargar el CSV
Standardized_dataset = df_final; % Usar la tabla ya en memoria

% Definimos las variables para PCA (las mismas que en Python)
columnas_pca = {'Poblacion', 'Superficie', 'Media_paro', 'Afiliados_SS', 'N_empresas', ...
                'Renta_disponible_media', 'Tasa_paro', 'Densidad_poblacion', ...
                'Transacciones_inmobiliarias', 'Deuda_ayuntamientos', 'Saldo_migratorio'};

% Verificar que todas las columnas PCA existen en Standardized_dataset
cols_exist_pca = ismember(columnas_pca, Standardized_dataset.Properties.VariableNames);
if ~all(cols_exist_pca)
    fprintf('Error: Las siguientes columnas requeridas para PCA no existen en la tabla estandarizada: %s\n', ...
            strjoin(columnas_pca(~cols_exist_pca), ', '));
    return; % Detener si faltan columnas
end

variables_table = Standardized_dataset(:, columnas_pca);
X = table2array(variables_table);

% Verificar si hay NaNs o Infs en X antes de PCA
if any(isinf(X(:))) || any(isnan(X(:)))
    warning('Se encontraron valores NaN o Inf en los datos (X) antes de PCA. Esto causará un error en PCA.');
    % Aquí deberías decidir cómo manejar estos valores. PCA no admite NaNs/Infs.
    % Opciones: eliminar filas, reimputar, etc.
    % Ejemplo: Eliminar filas con NaN/Inf (¡puede perder datos!)
    % valid_rows = all(~isnan(X) & ~isinf(X), 2);
    % X = X(valid_rows, :);
    % Standardized_dataset = Standardized_dataset(valid_rows, :); % Asegúrate de mantener la coherencia
    % fprintf('Se eliminaron %d filas con NaN/Inf antes de PCA.\n', sum(~valid_rows));
    % Si no se manejan, PCA fallará. Por ahora, solo advertimos.
    fprintf('Error: PCA no puede continuar con NaN/Inf. Revisa los datos en ''FINAL_DATOS_IMPUTADOS-2.csv'' o la estandarización.\n');
    return; % Detener la ejecución
end


% Usamos Codigo_Postal como etiquetas
try
    y = Standardized_dataset.Codigo_Postal;
    % Convertir a categórico para asegurar que gscatter lo trate como grupos distintos
    y_group = categorical(y);
    disp('Usando Codigo_Postal (categórico) para colorear los gráficos PCA.');
catch ME_cp
    warning('No se pudo usar Codigo_Postal como etiqueta: %s. Usando un solo color.');
    y_group = ones(size(X,1), 1); % Grupo único si falla
end


% Aplicamos PCA con 2 componentes
num_components_pca_viz = 2;
try
    [coeff, score, latent, ~, explained] = pca(X, 'NumComponents', num_components_pca_viz);
    X_new = score; % 'score' contiene las nuevas coordenadas (componentes principales)
catch ME_pca
    fprintf('Error durante la ejecución de PCA: %s\n', ME_pca.message);
    fprintf('Esto suele ocurrir si hay valores NaN o Inf en los datos de entrada X.\n');
    return; % Detener si PCA falla
end

% Graficar antes y después del PCA
figure('Name', 'PCA Comparación');
set(gcf, 'Position', [100, 100, 1000, 400]); % Ajustar tamaño ventana

% Antes del PCA (usando las 2 primeras variables originales estandarizadas)
subplot(1, 2, 1);
gscatter(X(:, 1), X(:, 2), y_group, [], '.', 10); % gscatter para colorear por grupo
xlabel('x1');
ylabel('x2');
title('Antes del PCA'); 
legend off; )
grid on;

% Después del PCA
subplot(1, 2, 2);
gscatter(X_new(:, 1), X_new(:, 2), y_group, [], '.', 10);
xlabel('PC1');
ylabel('PC2');
title('Después del PCA (2 Componentes)');
legend off;
grid on;

% Mostrar varianza explicada y componentes
fprintf('Varianza explicada por los %d componentes del PCA (%%):\n', num_components_pca_viz);
disp(explained'); % Porcentaje de varianza explicada
fprintf('Eigenvalues (varianza) de los %d componentes:\n', num_components_pca_viz);
disp(latent(1:num_components_pca_viz)'); % Varianza de cada componente
fprintf('Componentes Principales (loadings) para %d componentes:\n', num_components_pca_viz);
disp(abs(coeff)); % Coeficientes (loadings)


% CLUSTERING K-MEANS con PCA
fprintf('\n CLUSTERING K-MEANS con PCA \n');
% Usamos las mismas variables que para el PCA anterior (X ya está definido)

% 1. PCA para determinar número óptimo de componentes (Método del codo para varianza)
try
    [coeff_full, score_full, latent_full, ~, explained_full] = pca(X); % PCA con todos los componentes
catch ME_pca_full
     fprintf('Error durante la ejecución de PCA completo: %s\n', ME_pca_full.message);
     fprintf('Revisa si hay NaN/Inf en X.\n');
     return; % Detener si PCA falla
end

% *** CORRECCIÓN DEL ERROR YLIM ***
explained_full_prop = explained_full / 100; % Convertir a proporciones (0-1)
cum_explained_prop = cumsum(explained_full_prop); % Suma acumulada de proporciones

figure('Name', 'Varianza Explicada Acumulada por PCA');
set(gcf, 'Position', [150, 150, 800, 600]);
plot(1:length(explained_full_prop), cum_explained_prop, 'bo--', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Varianza Explicada Acumulada por Componentes PCA');
xlabel('Número de Componentes');
ylabel('Varianza Explicada Acumulada (Proporción)'); % Etiqueta corregida
grid on;

% Establecer límites de Y basados en las proporciones
y_lower_limit = min(cum_explained_prop); % El valor más bajo es el del primer componente
if isempty(y_lower_limit) % Manejar caso de datos vacíos/problemáticos
    y_lower_limit = 0;
end
ylim([y_lower_limit - 0.05, 1.05]); % Establece el límite inferior un poco por debajo del primero
xticks(1:length(explained_full_prop));

% 2. Aplicar PCA con el número de componentes elegido (según el codo, ~4)
num_components_pca_cluster = 4;
fprintf('Seleccionando %d componentes PCA para clustering basado en el gráfico.\n', num_components_pca_cluster);
% Usar los resultados del PCA completo ya calculado si es posible
if size(score_full, 2) >= num_components_pca_cluster
    scores_pca = score_full(:, 1:num_components_pca_cluster);
    coeff_sel = coeff_full(:, 1:num_components_pca_cluster); % Coeficientes correspondientes
else
    % Recalcular si el PCA inicial no tuvo suficientes componentes (no debería pasar aquí)
    [coeff_sel, scores_pca] = pca(X, 'NumComponents', num_components_pca_cluster);
end


% 3. Método del codo para K-Means sobre los datos reducidos por PCA (scores_pca)
max_k = 20;
wcss = zeros(1, max_k); % Within-cluster sum of squares
rng(42); % Para reproducibilidad de K-means

fprintf('Calculando WCSS para K-Means (k=1 a %d)...\n', max_k);
opts_kmeans = statset('Display','off'); % Para evitar mensajes de kmeans
try
    for k = 1:max_k
        % Usar 'plus' para inicialización k-means++
        % Aumentar réplicas para robustez
        [~, ~, sumd] = kmeans(scores_pca, k, 'Replicates', 10, 'MaxIter', 1000, 'Options', opts_kmeans, 'Start', 'plus');
        wcss(k) = sum(sumd); % Suma total de distancias intra-cluster al cuadrado
    end
catch ME_kmeans_wcss
    fprintf('Error calculando WCSS para k=%d: %s\n', k, ME_kmeans_wcss.message);
    fprintf('Puede indicar problemas con los datos en scores_pca.\n');
    return; % Detener si falla el cálculo del codo
end

figure('Name', 'Método del Codo para K-Means');
set(gcf, 'Position', [200, 200, 800, 600]);
plot(1:max_k, wcss, 'bo--', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Método del Codo para K-Means con PCA (WCSS)');
xlabel('Número de Clusters (k)');
ylabel('WCSS (Suma de cuadrados intra-cluster)');
grid on;
xticks(1:max_k);

% 4. Aplicar K-Means con el número de clusters elegido (según la regla del codo, ~4)
num_clusters = 4;
fprintf('Aplicando K-Means con k = %d clusters basado en el gráfico del codo.\n', num_clusters);
rng(42); % Resetear semilla para la ejecución final
try
    [idx, C] = kmeans(scores_pca, num_clusters, 'Replicates', 10, 'MaxIter', 1000, 'Display', 'off', 'Start', 'plus');
    % idx: índice del cluster para cada punto (1 a num_clusters)
    % C: coordenadas de los centroides en el espacio PCA
catch ME_kmeans_final
    fprintf('Error aplicando K-Means final: %s\n', ME_kmeans_final.message);
    return; % Detener si K-Means falla
end

% 5. Preparar datos para visualización y análisis de segmentos
% Usar Standardized_dataset que corresponde a las filas de X y scores_pca
variables_clustering_table = Standardized_dataset(:, columnas_pca); % Datos originales estandarizados
scores_pca_table = array2table(scores_pca, 'VariableNames', ...
    arrayfun(@(i) sprintf('Componente_%d', i), 1:num_components_pca_cluster, 'UniformOutput', false));

df_segm_pca_kmeans = [variables_clustering_table, scores_pca_table];
df_segm_pca_kmeans.Segment_Kmeans_PCA = idx; % Añadir índices numéricos del cluster

% Mapear índices numéricos (1, 2, 3, 4) a nombres de segmento
segment_map = {'first', 'second', 'third', 'fourth'}; % Nombres como en Python
df_segm_pca_kmeans.Segment = categorical(idx, 1:num_clusters, segment_map(1:num_clusters)); % Convertir a categórico con nombres

disp('Primeras filas de la tabla con segmentos y componentes PCA:');
disp(head(df_segm_pca_kmeans, 5));

% 6. Visualizar los clusters usando los dos primeros componentes PCA
figure('Name', 'Clusters K-Means sobre Componentes PCA');
set(gcf, 'Position', [250, 250, 800, 600]);
gscatter(df_segm_pca_kmeans.Componente_2, df_segm_pca_kmeans.Componente_1, df_segm_pca_kmeans.Segment, 'grmc', '.', 15);
xlabel('Componente 2');
ylabel('Componente 1');
title('Clusters por Componentes del PCA (K-Means)');
legend('Location', 'best');
grid on;

% 7. Mostrar centroides
disp('Centroides de los clusters (en el espacio PCA):');
disp(array2table(C, 'VariableNames', scores_pca_table.Properties.VariableNames)); % Mostrar como tabla

% INTERPRETACIÓN Y ANÁLISIS
fprintf('\n INTERPRETACIÓN Y ANÁLISIS \n');

% Asegurarse de que 'Municipio' esté disponible en Standardized_dataset
if ~ismember('Municipio', Standardized_dataset.Properties.VariableNames)
    warning('La columna "Municipio" no se encontró en el dataset estandarizado.');
else
    % Añadir la columna Municipio a la tabla de segmentos
    % Asegurarse de que las filas coincidan (deberían si no se eliminaron filas en PCA)
    df_segm_pca_kmeans = addvars(df_segm_pca_kmeans, Standardized_dataset.Municipio, 'Before', 1, 'NewVariableNames', 'Municipio');

    % Agrupar municipios por segmento
    segmentos_unicos = unique(df_segm_pca_kmeans.Segment, 'stable'); 
    fprintf('Municipios por Segmento:\n');
    for i = 1:length(segmentos_unicos)
        segmento_actual = segmentos_unicos(i);
        % Filtrar la tabla por el segmento actual
        municipios_en_segmento = df_segm_pca_kmeans.Municipio(df_segm_pca_kmeans.Segment == segmento_actual);
        fprintf('\nSegmento %s (%d municipios):\n', char(segmento_actual), numel(municipios_en_segmento)); % Convertir categórico a char para fprintf
        % Mostrar como lista (transponer para formato columna si son pocos)
        if numel(municipios_en_segmento) < 20
             disp(municipios_en_segmento);
        else % Mostrar solo los primeros si son demasiados
            disp('Primeros 20 municipios:');
            disp(head(municipios_en_segmento, 20));
        end
    end
end
