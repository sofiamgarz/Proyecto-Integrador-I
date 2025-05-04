% Script: monografia_analysis.m
% Descripción: Importa datos, imputa valores faltantes usando Python, normaliza,
% aplica PCA y clustering K-means en MATLAB.

%% Paso 1: Cargar datos originales (se usa solo para la imputación opcional)
dataFile_original = 'datasets/Monografia_final.csv';
T_original = readtable(dataFile_original);

% Mostrar primeras 5 filas del dataset original (informativo)
disp('Primeras 5 filas del dataset original (informativo):');
disp(T_original(1:5,:));

%% Paso 2: Imputación de datos faltantes con Python (se ejecuta pero su resultado T no se usa después)
disp('--- Ejecutando Paso 2: Imputación (el resultado se guardará pero no se usará en pasos siguientes) ---');
% Definir columnas a imputar
cols_to_impute = {'N_empresas','Renta_disponible_media','Tasa_paro', ...
                  'Densidad_poblacion','Transacciones_inmobiliarias', ...
                  'Deuda_ayuntamientos','Saldo_migratorio'};

% Extraer matriz de entrenamiento solo con las columnas a imputar
X_train_impute = T_original{:, cols_to_impute}; 

% Configurar entorno Python (asegúrate de tener scikit-learn 1.6.1 o superior)
pe = pyenv;
if pe.Version < "3.8"
error('Python >= 3.8 es requerido con scikit-learn 1.6.1 instalado.');
end

% Importar IterativeImputer directamente (ya no es experimental en 1.6.1)
iter_mod = py.importlib.import_module('sklearn.impute._iterative');
IterImp = iter_mod.IterativeImputer(pyargs('max_iter', int32(11), 'random_state', int32(0)));

% Convertir datos a lista de listas de Python
pyData = mat2pyList(X_train_impute);

% Ajustar imputador y transformar los datos
try
    imp_fit = IterImp.fit(pyData);
    disp('Transformando datos con IterativeImputer...');
    pyArray = imp_fit.transform(pyData);
    X_imp_values = double(py.array.array('d', py.numpy.nditer(pyArray)));
    X_imp_matrix = reshape(X_imp_values, size(X_train_impute));
catch ME
     error('Error durante la imputación con Python: %s', ME.message);
end

% Crear una tabla temporal T_imputada_paso2 para guardar el resultado
T_imputada_paso2 = T_original; % Copiar la tabla original
for j = 1:length(cols_to_impute)
    T_imputada_paso2.(cols_to_impute{j}) = X_imp_matrix(:,j);
end

% CORRECCIÓN (en la tabla temporal): Cambiar valores negativos de N_empresas a cero
negIdx = T_imputada_paso2.N_empresas < 0;
if any(negIdx)
    disp('Cambiando valores negativos de N_empresas a 0 (en tabla temporal)...');
    T_imputada_paso2.N_empresas(negIdx) = 0;
end

% Guardar CSV de imputación realizada en este paso (opcional)
writetable(T_imputada_paso2, 'datasets/datos_imputados_paso2_matlab.csv');
disp('Resultado de la imputación del Paso 2 guardado en datasets/datos_imputados_paso2_matlab.csv');
disp('--- Fin Paso 2 ---');

%% **** INICIO DE ANÁLISIS CON ARCHIVO PRE-IMPUTADO ****
% A partir de aquí, se ignora T_imputada_paso2 y se carga el archivo especificado

disp('--- Cargando archivo pre-imputado para análisis: FINAL_DATOS_IMPUTADOS-2.csv ---');
dataFile_imputed = 'datasets/FINAL_DATOS_IMPUTADOS-2.csv';
try
    T = readtable(dataFile_imputed); % Sobrescribir T con los datos del archivo imputado especificado
    disp('Archivo FINAL_DATOS_IMPUTADOS-2.csv cargado correctamente.');
    % Mostrar primeras 5 filas del dataset cargado
    disp('Primeras 5 filas del dataset cargado para análisis:');
    disp(T(1:5,:));
catch ME
    error('No se pudo cargar el archivo %s. Verifique que existe y está en la carpeta datasets. Error: %s', dataFile_imputed, ME.message);
end

%% Paso 3: Normalización estándar (Z-score) - Usando la tabla T cargada
disp('--- Iniciando Paso 3: Normalización ---');
% Seleccionar SOLO las columnas numéricas que participarán en PCA/Clustering
% (Asegúrate que estas columnas existen en FINAL_DATOS_IMPUTADOS-2.csv y son las deseadas)
cols_for_analysis = {'Poblacion', 'Superficie', 'Media_paro', 'Afiliados_SS', ...
                     'N_empresas', 'Renta_disponible_media', 'Tasa_paro', ...
                     'Densidad_poblacion','Transacciones_inmobiliarias', ...
                     'Deuda_ayuntamientos','Saldo_migratorio'};

% Verificar que todas las columnas existan en la tabla T cargada
missing_cols = setdiff(cols_for_analysis, T.Properties.VariableNames);
if ~isempty(missing_cols)
    error('Las siguientes columnas especificadas para análisis no existen en la tabla cargada: %s', strjoin(missing_cols, ', '));
end

X = T{:, cols_for_analysis}; % Extraer la matriz de datos para análisis desde la T cargada

% Aplicar normalización z-score
X_mean = mean(X);
X_std = std(X);
X_std(X_std == 0) = 1; % Evitar división por cero
Xz = (X - X_mean) ./ X_std;

% Reconstruir tabla estandarizada (solo con columnas relevantes + identificadores de T cargada)
Tz_analysis = array2table(Xz, 'VariableNames', cols_for_analysis);
% Asegurarse que Municipio y Codigo_Postal existen en la T cargada
if ~ismember('Municipio', T.Properties.VariableNames) || ~ismember('Codigo_Postal', T.Properties.VariableNames)
    error('La tabla cargada T debe contener las columnas Municipio y Codigo_Postal.');
end
Tz = addvars(Tz_analysis, T.Municipio, T.Codigo_Postal, 'Before', cols_for_analysis{1});

% Guardar CSV normalizado (opcional, basado en T cargada)
writetable(Tz, 'datasets/standardized_dataset_from_loaded_matlab.csv');
disp('Dataset estandarizado (desde T cargada) guardado en datasets/standardized_dataset_from_loaded_matlab.csv');

%% Paso 4: PCA - Usando datos de T cargada y normalizada
disp('--- Iniciando Paso 4: PCA ---');
Xpca = Xz; % Usar la matriz ya estandarizada

% Ejecutar PCA
[coeff, score, latent, tsquared, explained, mu] = pca(Xpca);

% --- Gráfica Antes y Después de PCA ---
figure('Name', 'PCA: Antes y Después (Datos Cargados)');
subplot(1,2,1);
scatter(Xpca(:,1), Xpca(:,2), 15, T.Codigo_Postal, 'filled');
title('Antes de PCA (2 Primeras Vars. Estandarizadas)');
xlabel(cols_for_analysis{1});
ylabel(cols_for_analysis{2});
grid on;

subplot(1,2,2);
scatter(score(:,1), score(:,2), 15, T.Codigo_Postal, 'filled');
title('Después de PCA (Primeros 2 Componentes)');
xlabel('Componente Principal 1 (PC1)');
ylabel('Componente Principal 2 (PC2)');
grid on;

% --- Gráfica de Varianza Explicada por Componentes ---
figure('Name', 'PCA: Varianza Explicada (Datos Cargados)');
subplot(1,2,1);
pareto(explained);
title('Varianza Explicada por Componente');
xlabel('Componente Principal');
ylabel('Varianza Explicada (%)');
grid on;

subplot(1,2,2);
cumulativeExplained = cumsum(explained);
plot(1:length(cumulativeExplained), cumulativeExplained, '-o');
title('Varianza Acumulada Explicada');
xlabel('Número de Componentes Principales');
ylabel('Varianza Acumulada Explicada (%)');
yline(90, '--r', '90% Varianza');
ylim([0 105]);
grid on;

% Mostrar tabla de varianza explicada
disp('Varianza explicada por componentes PCA (%):');
disp(table((1:length(explained))', explained, cumulativeExplained, 'VariableNames', {'Componente','VarianzaExplicada', 'VarianzaAcumulada'}));

%% Paso 5: Clustering K-means en espacio PCA - Usando datos de T cargada
disp('--- Iniciando Paso 5: K-Means Clustering ---');
% Seleccionar número de componentes
numC = 4;
score_k = score(:,1:numC);

% --- Método del codo para elegir k en K-means ---
max_k = 20;
wcss = zeros(max_k, 1);
disp('Calculando WCSS para el método del codo de K-means...');
opts = statset('Display','off');
for k = 1:max_k
    try
        [~, ~, sumd] = kmeans(score_k, k, 'Replicates', 5, 'Options', opts, 'MaxIter', 200);
        wcss(k) = sum(sumd);
    catch ME
        warning('Error en k-means para k=%d: %s. Saltando.', k, ME.message);
        wcss(k) = NaN;
    end
end
valid_k = 1:max_k; valid_wcss = wcss;
valid_k(isnan(wcss)) = []; valid_wcss(isnan(wcss)) = [];

figure('Name', 'K-Means: Método del Codo (Datos Cargados)');
plot(valid_k, valid_wcss, '--o');
title('Método del Codo para K-Means (en espacio PCA)');
xlabel('Número de Clusters (k)');
ylabel('Suma de Cuadrados Intra-cluster (WCSS)');
grid on;

% --- Aplicar k-means con k=4 ---
num_clusters = 4;
disp(['Aplicando K-Means con k = ', num2str(num_clusters), '...']);
rng(42); % Fijar semilla
[idx, C] = kmeans(score_k, num_clusters, 'Replicates', 10, 'MaxIter', 300);

% Añadir segmentos a la tabla final (basada en Tz que viene de T cargada)
Tfinal = Tz;
Tfinal.Segment = idx;
labels = {'first','second','third','fourth'}; % Ajustar si es necesario
segment_map = containers.Map(1:num_clusters, labels(1:num_clusters));
Tfinal.SegmentLabel = cell(height(Tfinal), 1);
for i = 1:height(Tfinal)
    if isKey(segment_map, Tfinal.Segment(i))
         Tfinal.SegmentLabel{i} = segment_map(Tfinal.Segment(i));
    else
         Tfinal.SegmentLabel{i} = 'unknown';
    end
end
Tfinal.SegmentLabel = categorical(Tfinal.SegmentLabel);

% --- Graficar clusters en el espacio de los primeros 2 PCs ---
figure('Name', 'K-Means: Clusters en PCA (Datos Cargados)');
gscatter(score_k(:,1), score_k(:,2), Tfinal.SegmentLabel); % PC1 vs PC2
title(['Clusters K-Means (k=', num2str(num_clusters), ') en Espacio PCA']);
xlabel('Componente Principal 1 (PC1)');
ylabel('Componente Principal 2 (PC2)');
grid on;
legend('Location', 'best');

% Mostrar centroides
disp(['Centroides del clustering K-Means (en espacio PCA de ', num2str(numC), ' dimensiones):']);
disp(C);

%% Paso 6: Interpretación - Usando datos de T cargada
disp('--- Iniciando Paso 6: Interpretación ---');
disp('--- Municipios por Segmento (basado en datos cargados) ---');
segment_categories = categories(Tfinal.SegmentLabel);
for s = 1:length(segment_categories)
    segName = segment_categories{s};
    segment_indices = (Tfinal.SegmentLabel == segName);
    mun = Tfinal.Municipio(segment_indices);
    max_mun_display = 10;
    if length(mun) > max_mun_display
        fprintf('Segmento %s (%d municipios): %s, ...\n', segName, length(mun), strjoin(mun(1:max_mun_display)','; '));
    else
        fprintf('Segmento %s (%d municipios): %s\n', segName, length(mun), strjoin(mun','; '));
    end
end

%% Funciones auxiliares (sin cambios)
function pyList = mat2pyList(mat)
    [rows, cols] = size(mat);
    pyList = py.list();
    for i = 1:rows
        row_list = py.list();
        for j = 1:cols
            val = mat(i,j);
            if isnan(val)
                row_list.append(py.None);
            else
                row_list.append(val);
            end
        end
        pyList.append(row_list);
    end
end