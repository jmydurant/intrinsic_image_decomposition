function umfpack_make
%UMFPACK_MAKE to compile umfpack2 for use in MATLAB
%
% Compiles the umfpack2 mexFunction and then runs a simple demo.
%
% Example:
%   umfpack_make
%
% UMFPACK relies on AMD and its own built-in version of COLAMD for its ordering
% options.  The default is for UMFPACK to also use CHOLMOD, CCOLAMD, CAMD, and
% METIS for more ordering options as well.  This results in lower fill-in and
% higher performance.  A copy of METIS 4.0.1 must be placed in ../../metis-4.0.
% METIS is optional; if not present, it is not used.
%
% See also: umfpack, umfpack2, umfpack_details, umfpack_report, umfpack_demo,
% and umfpack_simple.

% Copyright 1995-2009 by Timothy A. Davis.

metis_path = '../../metis-4.0' ;
with_cholmod = exist ([metis_path '/Lib'], 'dir') ;

details = 0 ;   % set to 1 to print out each mex command as it's executed

flags = '' ;
is64 = ~isempty (strfind (computer, '64')) ;
if (is64)
    flags = ' -largeArrayDims' ;
end

v = version ;
try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
    mac = ismac ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
    mac = 0 ;
end
fprintf ('Compiling UMFPACK for MATLAB Version %s\n', v) ;

if (pc)
    obj = 'obj' ;
else
    obj = 'o' ;
end

kk = 0 ;

%-------------------------------------------------------------------------------
% BLAS option
%-------------------------------------------------------------------------------

% This is exceedingly ugly.  The MATLAB mex command needs to be told where to
% fine the LAPACK and BLAS libraries, which is a real portability nightmare.

if (pc)
    if (verLessThan ('matlab', '6.5'))
        % MATLAB 6.1 and earlier: use the version supplied here
        lapack = 'lcc_lib/libmwlapack.lib' ;
        fprintf ('Using %s.  If this fails with dgemm and others\n',lapack);
        fprintf ('undefined, then edit umfpack_make.m and modify the') ;
        fprintf (' statement:\nlapack = ''%s'' ;\n', lapack) ;
    elseif (verLessThan ('matlab', '7.5'))
        lapack = 'libmwlapack.lib' ;
    else
        % MATLAB R2007b (7.5) made the problem worse
        lapack = 'libmwlapack.lib libmwblas.lib' ;
    end
else
    % For other systems, mex should find lapack on its own, but this has
    % been broken in MATLAB R2007a; the following is now required.
    if (verLessThan ('matlab', '7.5'))
        lapack = '-lmwlapack' ;
    else
        % MATLAB R2007b (7.5) made the problem worse
        lapack = '-lmwlapack -lmwblas' ;
    end
end

if (is64 && ~verLessThan ('matlab', '7.8'))
    % versions 7.8 and later on 64-bit platforms use a 64-bit BLAS
    fprintf ('with 64-bit BLAS\n') ;
    flags = [flags ' -DBLAS64'] ;
end

%-------------------------------------------------------------------------------
% -DNPOSIX option (for sysconf and times timer routines)
%-------------------------------------------------------------------------------

posix = ' ' ;
if (~ (pc || mac))
    % added for timing routine:
    lapack = [lapack ' -lrt'] ;
    posix = ' -DLIBRT' ;
end

%-------------------------------------------------------------------------------
% Source and include directories
%-------------------------------------------------------------------------------

umfdir = '../Source/' ;
amddir = '../../AMD/Source/' ;
incdir = ' -I. -I../Include -I../Source -I../../AMD/Include -I../../UFconfig' ;

if (with_cholmod)
    incdir = [incdir ' -I../../CCOLAMD/Include -I../../CAMD/Include -I../../CHOLMOD/Include -I' metis_path '/Lib -I../../COLAMD/Include'] ;
end

%-------------------------------------------------------------------------------
% METIS patch
%-------------------------------------------------------------------------------

% fix the METIS 4.0.1 rename.h file
if (with_cholmod)
    fprintf ('with CHOLMOD, CAMD, CCOLAMD, and METIS\n') ;
    f = fopen ('rename.h', 'w') ;
    if (f == -1)
        error ('unable to create rename.h in current directory') ;
    end
    fprintf (f, '/* do not edit this file; generated by umfpack_make.m */\n') ;
    fprintf (f, '#undef log2\n') ;
    fprintf (f, '#include "%s/Lib/rename.h"\n', metis_path) ;
    fprintf (f, '#undef log2\n') ;
    fprintf (f, '#define log2 METIS__log2\n') ;
    fprintf (f, '#include "mex.h"\n') ;
    fprintf (f, '#define malloc mxMalloc\n') ;
    fprintf (f, '#define free mxFree\n') ;
    fprintf (f, '#define calloc mxCalloc\n') ;
    fprintf (f, '#define realloc mxRealloc\n') ;
    fclose (f) ;
    flags = [' -DNSUPERNODAL -DNMODIFY -DNMATRIXOPS ' flags] ;
else
    fprintf ('without CHOLMOD, CAMD, CCOLAMD, and METIS\n') ;
    flags = [' -DNCHOLMOD ' flags] ;
end

%-------------------------------------------------------------------------------
% source files
%-------------------------------------------------------------------------------

% non-user-callable umf_*.[ch] files:
umfch = { 'assemble', 'blas3_update', ...
        'build_tuples', 'create_element', ...
        'dump', 'extend_front', 'garbage_collection', ...
        'get_memory', 'init_front', 'kernel', ...
        'kernel_init', 'kernel_wrapup', ...
        'local_search', 'lsolve', 'ltsolve', ...
        'mem_alloc_element', 'mem_alloc_head_block', ...
        'mem_alloc_tail_block', 'mem_free_tail_block', ...
        'mem_init_memoryspace', ...
        'report_vector', 'row_search', 'scale_column', ...
        'set_stats', 'solve', 'symbolic_usage', 'transpose', ...
        'tuple_lengths', 'usolve', 'utsolve', 'valid_numeric', ...
        'valid_symbolic', 'grow_front', 'start_front', ...
	'store_lu', 'scale' } ;

% non-user-callable umf_*.[ch] files, int versions only (no real/complex):
umfint = { 'analyze', 'apply_order', 'colamd', 'free', 'fsize', ...
        'is_permutation', 'malloc', 'realloc', 'report_perm', ...
	'singletons', 'cholmod' } ;

% non-user-callable and user-callable amd_*.[ch] files (int versions only):
amdsrc = { 'aat', '1', '2', 'dump', 'postorder', 'post_tree', 'defaults', ...
        'order', 'control', 'info', 'valid', 'preprocess', 'global' } ;

% user-callable umfpack_*.[ch] files (real/complex):
user = { 'col_to_triplet', 'defaults', 'free_numeric', ...
        'free_symbolic', 'get_numeric', 'get_lunz', ...
        'get_symbolic', 'get_determinant', 'numeric', 'qsymbolic', ...
        'report_control', 'report_info', 'report_matrix', ...
        'report_numeric', 'report_perm', 'report_status', ...
        'report_symbolic', 'report_triplet', ...
        'report_vector', 'solve', 'symbolic', ...
        'transpose', 'triplet_to_col', 'scale' ...
	'load_numeric', 'save_numeric', 'load_symbolic', 'save_symbolic' } ;

% user-callable umfpack_*.[ch], only one version
generic = { 'timer', 'tictoc', 'global' } ;

M = cell (0) ;

if (with_cholmod)

    % other source:

    camd_src = { ...
        '../../CAMD/Source/camd_1', ...
        '../../CAMD/Source/camd_2', ...
        '../../CAMD/Source/camd_aat', ...
        '../../CAMD/Source/camd_control', ...
        '../../CAMD/Source/camd_defaults', ...
        '../../CAMD/Source/camd_dump', ...
        '../../CAMD/Source/camd_global', ...
        '../../CAMD/Source/camd_info', ...
        '../../CAMD/Source/camd_order', ...
        '../../CAMD/Source/camd_postorder', ...
        '../../CAMD/Source/camd_preprocess', ...
        '../../CAMD/Source/camd_valid' } ;

    colamd_src = {
        '../../COLAMD/Source/colamd', ...
        '../../COLAMD/Source/colamd_global' } ;

    ccolamd_src = {
        '../../CCOLAMD/Source/ccolamd', ...
        '../../CCOLAMD/Source/ccolamd_global' } ;

    metis_src = {
        'Lib/balance', ...
        'Lib/bucketsort', ...
        'Lib/ccgraph', ...
        'Lib/coarsen', ...
        'Lib/compress', ...
        'Lib/debug', ...
        'Lib/estmem', ...
        'Lib/fm', ...
        'Lib/fortran', ...
        'Lib/frename', ...
        'Lib/graph', ...
        'Lib/initpart', ...
        'Lib/kmetis', ...
        'Lib/kvmetis', ...
        'Lib/kwayfm', ...
        'Lib/kwayrefine', ...
        'Lib/kwayvolfm', ...
        'Lib/kwayvolrefine', ...
        'Lib/match', ...
        'Lib/mbalance2', ...
        'Lib/mbalance', ...
        'Lib/mcoarsen', ...
        'Lib/memory', ...
        'Lib/mesh', ...
        'Lib/meshpart', ...
        'Lib/mfm2', ...
        'Lib/mfm', ...
        'Lib/mincover', ...
        'Lib/minitpart2', ...
        'Lib/minitpart', ...
        'Lib/mkmetis', ...
        'Lib/mkwayfmh', ...
        'Lib/mkwayrefine', ...
        'Lib/mmatch', ...
        'Lib/mmd', ...
        'Lib/mpmetis', ...
        'Lib/mrefine2', ...
        'Lib/mrefine', ...
        'Lib/mutil', ...
        'Lib/myqsort', ...
        'Lib/ometis', ...
        'Lib/parmetis', ...
        'Lib/pmetis', ...
        'Lib/pqueue', ...
        'Lib/refine', ...
        'Lib/separator', ...
        'Lib/sfm', ...
        'Lib/srefine', ...
        'Lib/stat', ...
        'Lib/subdomains', ...
        'Lib/timing', ...
        'Lib/util' } ;

    for i = 1:length (metis_src)
        metis_src {i} = [metis_path '/' metis_src{i}] ;
    end

    cholmod_src = {
        '../../CHOLMOD/Core/cholmod_aat', ...
        '../../CHOLMOD/Core/cholmod_add', ...
        '../../CHOLMOD/Core/cholmod_band', ...
        '../../CHOLMOD/Core/cholmod_change_factor', ...
        '../../CHOLMOD/Core/cholmod_common', ...
        '../../CHOLMOD/Core/cholmod_complex', ...
        '../../CHOLMOD/Core/cholmod_copy', ...
        '../../CHOLMOD/Core/cholmod_dense', ...
        '../../CHOLMOD/Core/cholmod_error', ...
        '../../CHOLMOD/Core/cholmod_factor', ...
        '../../CHOLMOD/Core/cholmod_memory', ...
        '../../CHOLMOD/Core/cholmod_sparse', ...
        '../../CHOLMOD/Core/cholmod_transpose', ...
        '../../CHOLMOD/Core/cholmod_triplet', ...
        '../../CHOLMOD/Check/cholmod_check', ...
        '../../CHOLMOD/Cholesky/cholmod_amd', ...
        '../../CHOLMOD/Cholesky/cholmod_analyze', ...
        '../../CHOLMOD/Cholesky/cholmod_colamd', ...
        '../../CHOLMOD/Cholesky/cholmod_etree', ...
        '../../CHOLMOD/Cholesky/cholmod_postorder', ...
        '../../CHOLMOD/Cholesky/cholmod_rowcolcounts', ...
        '../../CHOLMOD/Partition/cholmod_ccolamd', ...
        '../../CHOLMOD/Partition/cholmod_csymamd', ...
        '../../CHOLMOD/Partition/cholmod_camd', ...
        '../../CHOLMOD/Partition/cholmod_metis', ...
        '../../CHOLMOD/Partition/cholmod_nesdis' } ;

    other_source = [cholmod_src metis_src ccolamd_src camd_src colamd_src] ;
    % other_source = [other_source { '../User/umfpack_l_cholmod' }] ;         %#ok

else
    other_source = { } ;
end

if (pc)
    % Windows does not have drand48 and srand48, required by METIS.  Use
    % drand48 and srand48 in CHOLMOD/MATLAB/Windows/rand48.c instead.
    other_source = [other_source {'../../CHOLMOD/MATLAB/Windows/rand48'}] ;
    incdir = [incdir ' -I../../CHOLMOD/MATLAB/Windows'] ;
end

incdir = strrep (incdir, '/', filesep) ;

%-------------------------------------------------------------------------------
% mex command
%-------------------------------------------------------------------------------

% with optimization:
mx = sprintf ('mex -O%s%s%s ', posix, incdir, flags) ;
% no optimization:
%% mx = sprintf ('mex -g %s%s%s ', posix, incdir, flags) ;
fprintf ('compile options:\n%s\n', mx) ;

%----------------------------------------
% CHOLMOD, CAMD, CCOLAMD, and METIS
%----------------------------------------

if (with_cholmod)
    for k = 1:length(other_source)
        fs = strrep (other_source {k}, '/', filesep) ;
        slash = strfind (fs, filesep) ;
        slash = slash (end) + 1 ;
        o = fs (slash:end) ;
        kk = cmd (sprintf ('%s -DDLONG -c %s.c', mx, fs), kk, details) ;
        M {end+1} = [o '.' obj] ;
    end
end

%-------------------------------------------------------------------------------
% Create the umfpack2 and amd2 mexFunctions for MATLAB (int versions only)
%-------------------------------------------------------------------------------

for k = 1:length(umfint)
    [M, kk] = make (M, '%s -DDLONG -c %sumf_%s.c', 'umf_%s.%s', ...
	'umf_%s_%s.%s', mx, umfint {k}, umfint {k}, 'm', obj, umfdir, ...
	kk, details) ;
end

rules = { [mx ' -DDLONG'] , [mx ' -DZLONG'] } ;
kinds = { 'md', 'mz' } ;

for what = 1:2

    rule = rules {what} ;
    kind = kinds {what} ;

    [M, kk] = make (M, '%s -DCONJUGATE_SOLVE -c %sumf_%s.c', 'umf_%s.%s', ...
        'umf_%s_%s.%s', rule, 'ltsolve', 'lhsolve', kind, obj, umfdir, ...
	kk, details) ;

    [M, kk] = make (M, '%s -DCONJUGATE_SOLVE -c %sumf_%s.c', 'umf_%s.%s', ...
        'umf_%s_%s.%s', rule, 'utsolve', 'uhsolve', kind, obj, umfdir, ...
	kk, details) ;

    [M, kk] = make (M, '%s -DDO_MAP -c %sumf_%s.c', 'umf_%s.%s', ...
        'umf_%s_%s_map_nox.%s', rule, 'triplet', 'triplet', kind, obj, ...
	umfdir, kk, details) ;

    [M, kk] = make (M, '%s -DDO_VALUES -c %sumf_%s.c', 'umf_%s.%s', ...
        'umf_%s_%s_nomap_x.%s', rule, 'triplet', 'triplet', kind, obj, ...
	umfdir, kk, details) ;

    [M, kk] = make (M, '%s -c %sumf_%s.c', 'umf_%s.%s',  ...
        'umf_%s_%s_nomap_nox.%s', rule, 'triplet', 'triplet', kind, obj, ...
	umfdir, kk, details) ;

    [M, kk] = make (M, '%s -DDO_MAP -DDO_VALUES -c %sumf_%s.c', 'umf_%s.%s', ...
        'umf_%s_%s_map_x.%s', rule, 'triplet', 'triplet', kind, obj, ...
	umfdir, kk, details) ;

    [M, kk] = make (M, '%s -DFIXQ -c %sumf_%s.c', 'umf_%s.%s', ...
	'umf_%s_%s_fixq.%s', rule, 'assemble', 'assemble', kind, obj, ...
	umfdir, kk, details) ;

    [M, kk] = make (M, '%s -DDROP -c %sumf_%s.c', 'umf_%s.%s', ...
	'umf_%s_%s_drop.%s', rule, 'store_lu', 'store_lu', kind, obj, ...
	umfdir, kk, details) ;

    for k = 1:length(umfch)
        [M, kk] = make (M, '%s -c %sumf_%s.c', 'umf_%s.%s', 'umf_%s_%s.%s', ...
            rule, umfch {k}, umfch {k}, kind, obj, umfdir, kk, details) ;
    end

    [M, kk] = make (M, '%s -DWSOLVE -c %sumfpack_%s.c', 'umfpack_%s.%s', ...
        'umfpack_%s_w%s.%s', rule, 'solve', 'solve', kind, obj, umfdir, ...
	kk, details) ;

    for k = 1:length(user)
        [M, kk] = make (M, '%s -c %sumfpack_%s.c', 'umfpack_%s.%s', ...
            'umfpack_%s_%s.%s', rule, user {k}, user {k}, kind, obj, ...
	    umfdir, kk, details) ;
    end
end

for k = 1:length(generic)
    [M, kk] = make (M, '%s -c %sumfpack_%s.c', 'umfpack_%s.%s', ...
	'umfpack_%s_%s.%s', mx, generic {k}, generic {k}, 'm', obj, ...
	umfdir, kk, details) ;
end

%----------------------------------------
% AMD routines (long only)
%----------------------------------------

for k = 1:length(amdsrc)
    [M, kk] = make (M, '%s -DDLONG -c %samd_%s.c', 'amd_%s.%s', ...
	'amd_%s_%s.%s', mx, amdsrc {k}, amdsrc {k}, 'm', obj, amddir, ...
	kk, details) ;
end

%----------------------------------------
% compile the umfpack2 mexFunction
%----------------------------------------

C = sprintf ('%s -output umfpack2 umfpackmex.c', mx) ;
for i = 1:length (M)
    C = [C ' ' (M {i})] ;   %#ok
end
C = [C ' ' lapack] ;
kk = cmd (C, kk, details) ;

%----------------------------------------
% delete the object files
%----------------------------------------

for i = 1:length (M)
    rmfile (M {i}) ;
end

%----------------------------------------
% compile the luflop mexFunction
%----------------------------------------

cmd (sprintf ('%s -output luflop luflopmex.c', mx), kk, details) ;

fprintf ('\nUMFPACK successfully compiled\n') ;

%===============================================================================
% end of umfpack_make
%===============================================================================


%-------------------------------------------------------------------------------

function rmfile (file)
% rmfile:  delete a file, but only if it exists
if (length (dir (file)) > 0)						    %#ok
    delete (file) ;
end

%-------------------------------------------------------------------------------

function cpfile (src, dst)
% cpfile:  copy the src file to the filename dst, overwriting dst if it exists
rmfile (dst)
if (length (dir (src)) == 0)	%#ok
    fprintf ('File does not exist: %s\n', src) ;
    error ('File does not exist') ;
end
try
    copyfile (src, dst) ;
catch ME
    % ignore errors of the form "cp: preserving permissions: ...
    % Operation not supported".  rethrow all other errors.
    if (isempty (strfind (ME.message, 'Operation not supported')))
        rethrow (ME) ;
    end
end

%-------------------------------------------------------------------------------

function mvfile (src, dst)
% mvfile:  move the src file to the filename dst, overwriting dst if it exists
cpfile (src, dst) ;
rmfile (src) ;

%-------------------------------------------------------------------------------

function kk = cmd (s, kk, details)
%CMD: evaluate a command, and either print it or print a "."
if (details)
    fprintf ('%s\n', s) ;
else
    if (mod (kk, 60) == 0)
	fprintf ('\n') ;
    end
    kk = kk + 1 ;
    fprintf ('.') ;
end
eval (s) ;

%-------------------------------------------------------------------------------

function [M, kk] = make (M, s, src, dst, rule, file1, file2, kind, obj, ...
    srcdir, kk, details)
% make:  execute a "make" command for a source file
kk = cmd (sprintf (s, rule, srcdir, file1), kk, details) ;
src = sprintf (src, file1, obj) ;
dst = sprintf (dst, kind, file2, obj) ;
mvfile (src, dst) ;
M {end + 1} = dst ;
