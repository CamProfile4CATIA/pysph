<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>\
    ${tempotral_collection(files,times,particles_arrays)}\
  </Domain>
</Xdmf>

<%def name="tempotral_collection(files,times,particles_arrays)">
    <Grid Name="temporal" GridType="Collection" CollectionType="Temporal" >
    % for file,time in zip(files,times):
      <Grid Name="spatial" GridType="Collection" CollectionType="Spatial" >
        <Time Type="Single" Value="   ${time}" />\
        ${spatial_collection(file,particles_arrays)}\
      </Grid>
    % endfor
    </Grid>
</%def>

<%def name="spatial_collection(file,particles_arrays)">
  % for pname, data in particles_arrays.items():
        <Grid Name="${pname}" GridType="Uniform">\
          ${topo_and_geom(file, pname, data['n_particles'])}\
          ${variables_data(file, pname, data['n_particles'], data['output_props'],data['stride'], data['attr_type'])}\
        </Grid>
  % endfor
</%def>

<%def name="topo_and_geom(file, pname, n_particles)">
          <Topology TopologyType="Polyvertex" Dimensions="${n_particles}" NodesPerElement="1"/>
          <Geometry Type="X_Y_Z">
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/x
            </DataItem>
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/y
            </DataItem>
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/z
            </DataItem>
          </Geometry>
</%def>

<%def name="variables_data(file, pname, n_particles, var_names, stride, attr_type)">
  % for var_name in var_names:
          <Attribute Name="${var_name}" AttributeType="${attr_type[var_name]}" Center="Node">
            <DataItem Format="HDF" Dimensions="${n_particles} ${stride[var_name]}" NumberType="float">
              ${file}:/particles/${pname}/arrays/${var_name}
            </DataItem>
          </Attribute>
  % endfor
</%def>