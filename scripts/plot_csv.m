% ============================================================================
% plot_csv.m
%
% Plots the CSV files written by csv_writer against the load factor: for
% each face, one figure of the mean displacement u and one of the mean
% nominal traction t = F / A_0, with one panel per reported direction. The
% result folders listed below, e.g. of driver_static_disp and driver_static_mixed,
% are overlaid in the same panels.
%
% Author: Chongran Zhao
% Date: Sep. 30, 2026
% Email: chongran_zhao@brown.edu
% ============================================================================
clear; close all; clc;

% Result folders, relative to the repository, and their legend labels.
csv_dirs = {'build_disp/results_csv', 'build_mixed/results_csv', 'build_disp_dense/results_csv', 'build_mixed_dense/results_csv'};
labels = {'displacement', 'mixed', 'displacement_dense', 'mixed_dense'};

assert(numel(labels) == numel(csv_dirs), 'One label per folder.');

% The repository is the parent of the folder of this script.
repo_dir = fileparts(fileparts(mfilename('fullpath')));
csv_dirs = fullfile(repo_dir, csv_dirs);

% The faces are those of the first folder.
files = dir(fullfile(csv_dirs{1}, '*.csv'));
assert(~isempty(files), 'No CSV file in %s; run csv_writer first.', csv_dirs{1});

for ff = 1:numel(files)
   [~, face] = fileparts(files(ff).name);

   results = cell(1, numel(csv_dirs));
   for dd = 1:numel(csv_dirs)
      csv_file = fullfile(csv_dirs{dd}, files(ff).name);
      assert(isfile(csv_file), 'Missing %s.', csv_file);
      results{dd} = readtable(csv_file);
   end

   % Reported directions, from the columns u_x, u_y, u_z.
   columns = results{1}.Properties.VariableNames;
   dirs = erase(columns(startsWith(columns, 'u_')), 'u_');

   % One marker per folder, so that the curves stay apart where they overlap.
   markers = {'o', 's', '^', 'd', 'v', 'x', '+', '*'};

   % One figure per quantity: u, then t.
   quantities = {'u', 't'};
   for qq = 1:numel(quantities)
      figure('Name', sprintf('%s_%s', face, quantities{qq}));
      tiledlayout(1, numel(dirs));
      for kk = 1:numel(dirs)
         column = [quantities{qq} '_' dirs{kk}];

         nexttile;
         hold on;
         for dd = 1:numel(results)
            marker = markers{mod(dd - 1, numel(markers)) + 1};
            plot(results{dd}.load_factor, results{dd}.(column), ['-' marker], ...
                 'DisplayName', labels{dd});
         end
         hold off;
         box on;
         grid on;

         xlabel('load factor');
         ylabel(column, 'Interpreter', 'none');
         title(sprintf('%s, %s', face, column), 'Interpreter', 'none');
         legend('Location', 'best', 'Interpreter', 'none');
      end
   end
end
