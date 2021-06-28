
var spec_choropleth_t0 = "http://localhost:8000/js/choropleth.json";

vegaEmbed('#choropleth-t0', spec_choropleth_t0).then(function(result) {
}).catch(console.error);




var spec_choropleth_epi_state = "http://localhost:8000/js/choropleth-epi-state.json";

vegaEmbed('#choropleth-state', spec_choropleth_epi_state).then(function(result) {
}).catch(console.error);




var spec_time_series_stringency = "http://localhost:8000/js/time-series-stringency.json";

vegaEmbed('#time-series-stringency', spec_time_series_stringency).then(function(result) {
}).catch(console.error);


var spec_time_series_mean_stringency = "http://localhost:8000/js/time-series-mean-stringency.json";

vegaEmbed('#time-series-mean-stringency', spec_time_series_mean_stringency).then(function(result) {
}).catch(console.error);

var spec_time_vs_confirmed_scatter = "http://localhost:8000/js/time-vs-confirmed-scatter.vl.json";

vegaEmbed('#time-vs-confirmed-scatter', spec_time_vs_confirmed_scatter).then(function(result) {
}).catch(console.error);

