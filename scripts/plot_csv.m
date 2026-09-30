% ============================================================================
% plot_csv.m
%
% Plots the CSV files written by csv_writer, one figure per face and one
% panel per reported direction: the resultant force F against the mean
% displacement u, or against the load factor on a face whose u stays zero,
% e.g. a fixed one. The result folders listed below, e.g. of
% driver_displacement and driver_mixed, are overlaid in the same panels.
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

   figure('Name', face);
   tiledlayout(1, numel(dirs));
   for kk = 1:numel(dirs)
      disp_name = ['u_' dirs{kk}];
      force_name = ['F_' dirs{kk}];

      % On a face that does not move, F is plotted against the load factor.
      is_fixed = all(results{1}.(disp_name) == 0);

      nexttile;
      hold on;
      for dd = 1:numel(results)
         if is_fixed
            x_data = results{dd}.load_factor;
         else
            x_data = results{dd}.(disp_name);
         end
         plot(x_data, results{dd}.(force_name), 'o-', 'DisplayName', labels{dd});
      end
      hold off;
      box on;
      grid on;

      if is_fixed
         xlabel('load factor');
      else
         xlabel(disp_name, 'Interpreter', 'none');
      end
      ylabel(force_name, 'Interpreter', 'none');
      title(sprintf('%s, %s', face, dirs{kk}), 'Interpreter', 'none');
      legend('Location', 'best', 'Interpreter', 'none');
   end
end
