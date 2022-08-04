""" Generate XDMF file(s) referencing the heavy data stored using HDF5 by
PySPH.

Separate xdmf file will be generated for each hdf5 file input. If directory is
input, a single xdmf file will be generated assuming all the hdf5 files inside
the directory as timeseris data.
"""

import argparse
import sys
from pathlib import Path
import h5py
from mako.template import Template

from pysph.solver.utils import get_files, load


def main(argv=None):
    """ Main function to generate XDMF file(s) referencing the heavy data
    stored using HDF5 by PySPH.
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='generate_xdmf', description=__doc__, add_help=False
    )

    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )

    parser.add_argument(
        "inputfile", type=str, nargs='+',
        help="list of input hdf5 file(s) or/and director(y/ies) with hdf5"
             "file(s)."
    )

    parser.add_argument(
        "-d", "--outdir", metavar="outdir", type=str,
        default=Path(),
        help="directory to output xdmf file(s), defaults to current working "
             "directory"
    )

    parser.add_argument(
        "--relative-path", action="store_true",
        help="use relative path(s) to reference heavy data in the generated"
             " xdmf file"
    )

    parser.add_argument(
        "--vectorize-velocity", action="store_true",
        help="reference u,v and w such that the velocity is read as vector "
             "quantity through xdmf"
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    run(options)


def run(options):
    for ifile in options.inputfile:
        if Path(ifile).is_dir():
            files = get_files(ifile, endswith='hdf5')
            outfile = Path(ifile).name + '.xdmf'
        else:
            files = [ifile]
            outfile = Path(ifile).stem + '.xdmf'

        files2xdmf(files, Path(options.outdir).joinpath(outfile),
                   options.relative_path, options.vectorize_velocity)


def files2xdmf(absolute_files, outfilename, refer_relative_path,
               vectorize_velocity):
    times = []
    for fname in absolute_files:
        data = h5py.File(fname, 'r')  # will fail here if not .hdf5 file
        times.append(data['solver_data'].attrs.get('t'))

    pa = load(absolute_files[0])
    particles_arrays = {}
    for particles_name in pa.get('arrays').keys():
        particles = pa['arrays'].get(particles_name)
        output_props = particles.output_property_arrays
        if vectorize_velocity:
            for component in {'u', 'v', 'w'}:
                output_props.remove(component)
        n_particles = particles.num_real_particles
        _stride = particles.stride
        attr_type = {}
        stride = {}
        for var_name in output_props:
            stride[var_name] = _stride[var_name] if var_name in _stride else 1
            if stride[var_name] == 1:
                typ = 'Scalar'
            elif stride[var_name] == 3:
                typ = 'Vector'
            elif stride[var_name] == 9:
                typ = 'Tensor'
            else:
                typ = 'Matrix'
            attr_type[var_name] = typ

        particles_arrays[particles_name] = {'output_props': output_props,
                                            'n_particles': n_particles,
                                            'stride': stride,
                                            'attr_type': attr_type}

    template_file = Path(__file__).parent.absolute().joinpath(
        'xdmf_template.mako')
    xdmf_template = Template(filename=str(template_file))

    if refer_relative_path:
        outdir = Path(outfilename).parent
        files = [Path(f).relative_to(outdir) for f in absolute_files]
    else:
        files = absolute_files

    with open(outfilename, 'w') as xdmf_file:
        print(xdmf_template.render(times=times, files=files,
                                   particles_arrays=particles_arrays,
                                   vectorize_velocity=vectorize_velocity),
              file=xdmf_file)


if __name__ == '__main__':
    main()
