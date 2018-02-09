//
// Config
//
var chart_categories = [
    {
        title: 'Topology',
        slug: 'topology',
    },
    {
        title: 'DAP Results',
        slug: 'dap-results',
    },
    {
        title: 'DAP Metrics',
        slug: 'dap-metrics',
    },
];


//
// Globals
//
var plot_funcs = {
    line: _show_line,
    step_curve: _show_line,
    stacked_bars: _show_stacked_bars,
    scatter: _show_scatter_plot,
    topology: _show_topology,
};
var plotter = Plotter();
plotter.init(data);

//
// Functions
//
function Plotter() {
    var self = {};

    function init(data) {
        self.data = data;

        chart_categories.forEach(function(d, i) {
            var cat_title = d.title;
            var cat = d.slug;
            self[cat] = {};
            sheet = self[cat];

            _create_html_skeleton(sheet, cat_title, cat, data, i);
            if (data[cat].legend === true) {
                _show_legend(sheet, cat);
            }
            _show_enabled_charts(cat);
        });
    }

    function toggle(curve, show) {
        var parts = curve.split('.');
        var cat = parts[0];
        var item = parts[1];
        var item_data = self.data[cat].items[item];
        var sheet = self[cat];
        if (show) {
            var plot_func = plot_funcs[item_data.type];
            if (plot_func === null) {
                console.log('Unkown chart type. ' + curve);
            } else {
                plot_func(sheet, item_data, item);
            }
        } else {
            sheet.svg.select('.' + item).remove();
        }
    }

    function _create_html_skeleton(sheet, cat_title, cat, data, i) {
        var li = d3.select('header nav ul').append('li')
            .classed(cat, true)
            .classed('active', i == 0);
        li.append('a')
            .attr('href', '#' + cat)
            .text(cat_title)
            .on('click', function() {
                var target = this.href.split('#')[1];
                var nav_boxes = d3.selectAll('header nav li');
                var content_boxes = d3.selectAll('section.content');
                nav_boxes.classed('active', false);
                content_boxes.classed('active', false);
                nav_boxes.filter('.' + target).classed('active', true);
                content_boxes.filter('.' + target).classed('active', true);
            });

        var sec = d3.select('div.wrapper').insert('section', 'footer.footer')
            .classed('content', true)
            .classed(cat, true)
            .classed('active', i == 0)
        var svg = sec.append('svg').classed('canvas', true);
        var subsec = sec.append('section').classed('settings', true);
        subsec.append('h1').text('Settings');

        sheet.sec = sec;
        sheet.svg = svg;

        if (i == 0) {
            // Set total width and height for all charts
            self.width = parseInt(sheet.svg.style('width'), 10);
            self.height = parseInt(sheet.svg.style('height'), 10);
        }

        var figure = prepare_plot(sheet.svg, self.width, self.height,
                                 cat_title + ' ' + data.title,
                                 data[cat].x_axis, data[cat].y_axis)
        sheet.graph = figure.plot;
        sheet.w = figure.w;
        sheet.h = figure.h;
    }

    function _show_legend(sheet, cat) {
        // Data for the legend's svg images
        var legend_data = {
            line: {dx: [0, 100], dy: [0, 100],
                   data: [{x: 10, y: 50}, {x: 90, y: 50}]},
            stacked_bars: {dx: [-1, 2], dy: [-5, 40],
                           data: [[{v: 20, y0: 0}], [{v: 15, y0: 20}]]},
            scatter: {dx: [0, 100], dy: [0, 100],
                      data: [{x: 27, y: 50, class: 'gray_noalpha'},
                             {x: 73, y: 50, class: 'green_noalpha'}]},
            topology: {dx: null, dy: null,
                          data: {nodes: [{name: '1'}], links: []}},
        };
        legend_data.step_curve = legend_data.line;

        // Append the legend with checkboxes to select charts
        // subsec.append('div').classed('settings-legend', true);
        var fs = sheet.sec.select('.settings')
            .append('div')
            .append('fieldset');
        fs.append('legend').text('Show:');
        data[cat].item_names.forEach(function(d) {
            var title = data[cat].items[d].title;
            var chart_data = data[cat].items[d]

            var l = fs.append('label')
                .classed(d, true);
            l.append('input')
                .attr('type', 'checkbox')
                .attr('name', 'show_curve')
                .attr('value', cat + '.' + d)
                .attr('title', title)
                .on('change', function() {
                    toggle(this.value, this.checked);
                });
            var svg = l.append('svg')
                .classed('legend', true);
            l.append('span')
                .text(title);
            fs.append('br');

            var svg_sheet = {
                graph: svg,
                w: parseInt(svg.style('width'), 10),
                h: parseInt(svg.style('height'), 10),
            };
            var svg_data = {
                class: chart_data.class,
                color_range: chart_data.color_range,
                n_layers: 2,
                tooltips: [''],
                domain_x: legend_data[chart_data.type].dx,
                domain_y: legend_data[chart_data.type].dy,
                data: legend_data[chart_data.type].data,
            };
            var plot_func = plot_funcs[chart_data.type];
            if (plot_func === null) {
                console.log('Unkown chart type. ' + curve);
            } else {
                plot_func(svg_sheet, svg_data, cat + '.' + d);
            }
        });
    }

    function _show_enabled_charts(cat) {
        self.data[cat].enabled.forEach(function(item) {
            d3.select('section.' + cat + ' .settings .' + item + ' input')
                .property('checked', true);
            toggle(cat + '.' + item, true);
        });
    }

    return {
        init: init,
        toggle: toggle
    };
}

function _show_line(sheet, data, item) {
    var x = scale(data.domain_x, [0, sheet.w]);
    var y = scale(data.domain_y, [sheet.h, 0]);
    var line = d3.svg.line()
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); });

    sheet.graph.append('path')
        .datum(data.data)
        .attr('class', 'line ' + item)
        .attr('d', line)
        .classed(data.class, true);
}

function _show_stacked_bars(sheet, data, item) {
    var x = scale(data.domain_x, [0, sheet.w]);
    var y = scale(data.domain_y, [sheet.h, 0]);
    var color = scale([0, data.n_layers - 1], data.color_range);

    var group = sheet.graph.append('g')
        .attr('class', item);
    var layer = group.selectAll('.layer')
        .data(data.data)
        .enter().append('g')
            .attr('class', 'layer')
            .style('fill', function(d, i) { return color(i); });

    var rect = layer.selectAll('rect')
        .data(function(d) { return d; })
    .enter().append('rect')
        .attr('x',      function(d, i) { return x(i); })
        .attr('width',  1 * (x(1) - x(0)))
        .attr('y',      function(d) { return y(d.y0 + d.v); })
        .attr('height', function(d) { return y(d.y0) - y(d.y0 + d.v); });

    layer.each(function(d, i) {
        var rects = d3.select(this).selectAll('rect');
        // rects.append('title').text(data.tooltips[i]);
        rects.append('title').text(function(bar) {
            return data.tooltips[i] +
                ' (' + Math.round(bar.v * 10) / 10 + 'kW)';
        });
        rects.on('mouseover', function(d) {
            d3.selectAll(this.parentNode.childNodes)
                .classed('highlight', true);
        });
        rects.on('mouseout', function(d) {
            d3.selectAll(this.parentNode.childNodes)
                .classed('highlight', false);
        });
    });

}

function _show_scatter_plot(sheet, data, item) {
    var x = scale(data.domain_x, [0, sheet.w]);
    var y = scale(data.domain_y, [sheet.h, 0]);

    var group = sheet.graph.append('g')
        .attr('class', item);
    group.selectAll('circle').data(data.data)
        .enter().append('circle')
            .attr('class', function(d) { return d.class; })
            .attr('r', 3)
            .attr('cx', function(d) { return x(d.x); })
            .attr('cy', function(d) { return y(d.y); });
}

function _show_topology(sheet, data, item) {
    sheet.sec.select('.settings').append('div')
        .append('a')
        .on('click', msg_flow)
        .text('Show messages');

    var middle = Math.round(sheet.w / 2);
    var topo_pane = sheet.graph.append('g')
        .attr('class', item)
        .attr('transform', 'translate(0, 0)')
        .attr('width', middle)
        .attr('height', sheet.h);
    var chart_pane = sheet.graph.append('g')
        .attr('class', item)
        .attr('transform', 'translate(' + middle + ', 0)')
        .attr('width', sheet.w - middle)
        .attr('height', sheet.h);

    make_topo(topo_pane);
    var perf_figure = prepare_chart(chart_pane);

    function make_topo(group) {
        // Initialize force layout
        var force = d3.layout.force()
            .size([group.attr('width'), group.attr('height')])
            .links(data.data.links)
            .nodes(data.data.nodes)
            .friction(0.9)
            .linkDistance(function(d) { return d.length; })
            .linkStrength(function(d) { return d.strength; })
            .charge(-750)
            .gravity(0.15)
            .on('tick', tick)
            .start();

        // Add circle and line elements for the topology
        var links = group.selectAll('.link')
            .data(data.data.links);
        links.enter().append('line')
            .attr('class', 'link')
            .attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });

        var nodes = group.selectAll('.node')
            .data(data.data.nodes);
        nodes.enter().append('circle')
            .attr('class', function(d) { return 'node a' + d.name; })
            .attr('cx', function(d) { return d.x; })
            .attr('cy', function(d) { return d.y; })
            .attr('r', 10)
            .call(force.drag);
        nodes.append('title').text(function(node) { return node.name; });

        // Update the position of the SVG elements (circles and lines) at each
        // tick of the force simulation.
        function tick() {
            links
                .attr('x1', function(d) { return d.source.x; })
                .attr('y1', function(d) { return d.source.y; })
                .attr('x2', function(d) { return d.target.x; })
                .attr('y2', function(d) { return d.target.y; });

            nodes
                .attr('cx', function(d) { return d.x; })
                .attr('cy', function(d) { return d.y; });
        }
    }

    function prepare_chart(group) {
        var figure = prepare_plot(group,
                group.attr('width'), group.attr('height'),
                'Performance',
                {range: data.data.x_range, label: 'time', unit: 's'},
                {range: data.data.y_range, label: 'performance'});
        figure.title.classed('subtitle', true);
        return figure;
    }

    function msg_flow() {
        // Ten-times slower
        var speed = 1.5;  // msg speed (bezier computation)
        var msg_time_factor = 10;  // Speed-up for actual run-time
        // Real-time
        // var speed = 5;
        // var msg_time_factor = 1;

        var msg_data = data.data.msg_data;
        var msgs = [];

        var perf_x = scale(data.data.x_range, [0, perf_figure.w]);
        var perf_y = scale(data.data.y_range, [perf_figure.h, 0]);
        var perf_plot = perf_figure.plot;
        var perf_data = []

        var last = 0;
        d3.timer(function(elapsed) {
            elapsed /= 1000;

            // Remove received messages
            while (msgs.length && msgs[0].t >= 1) {
                msgs.shift();
            }
            // Move arriving messages from queue to msg list
            while (msg_data.length &&
                    msg_data[0].t * msg_time_factor <= elapsed) {
                var mdata = msg_data.shift();

                perf_data.push({x: mdata.t, y: mdata.perf, class: mdata.class});
                perf_plot.selectAll('circle').data(perf_data)
                    .enter().append('circle')
                        .attr('class', function(d) { return d.class; })
                        .attr('r', 0)
                        .attr('cx', function(d) { return perf_x(d.x); })
                        .attr('cy', function(d) { return perf_y(d.y); })
                        .transition()
                            .attr('r', 3);

                mdata.msgs.forEach(function(msg) {
                    var p0 = topo_pane.select('.node.a' + msg.s).datum();
                    var p2 = topo_pane.select('.node.a' + msg.d).datum();

                    // var p1 = get_ctrl_point(p0, p2);
                    msgs.push({
                        id: msg.id,
                        t_start: mdata.t,  // time in seconds when message got sent
                        t: 0,  // t value for bezier curve
                        // p: [p0, p1, p2],  // bezier control points
                        p: [p0, p2],  // bezier control points
                        b: null  // bezier point
                    });
                });
            }

            var dt = (elapsed - last) * speed;
            last = elapsed;
            update(dt);

            return (msgs.length === 0 && msg_data.length === 0);
        });

        function update(dt) {
            msgs.forEach(function(m) {
                m.t = Math.min(1, m.t + dt);
                m.b = get_b(m.p, m.t);
            });
            // Update data for circle
            var circle = topo_pane.selectAll('circle.msg')
                .data(msgs, function(m) { return m.id; });
            // Draw new circle if necessary
            circle.enter().append('circle')
                .attr('class', 'msg')
                // .attr('r', 0)
                // .transition(50)
                    .attr('r', 5);
            circle.exit()
                // .transition(50)
                //     .attr('r', 0)
                    .remove();
            // Update circle coordinates at each step
            circle
                .attr('cx', x)
                .attr('cy', y);
        }

        /**
        * Return a control point for source *s* and target *t*.
        */
        function get_ctrl_point(s, t) {
            return {x: t.x, y: s.y};
        }

        /**
        * Return the bezier point B for *t* and the control points *p*.
        */
        function get_b(p, t) {
            var levels = [p];
            for (var i = 1; i < p.length; i ++) {
                levels.push(interpolate(levels[i - 1], t));
            }
            var point = levels[levels.length - 1][0];
            return point;
        }

        /**
        * Interpolate the intermediate points for each level (except the first one).
        */
        function interpolate(p, t) {
            var r = [];
            for (var i = 1; i < p.length; i ++) {
                var p0 = p[i-1], p1 = p[i];
                r.push({x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t});
            }
            return r;
        }

        function x(m) { return m.b.x; }
        function y(m) { return m.b.y; }
    }
}

function scale(domain, range) {
    var scale = d3.scale.linear()
        .domain(domain)
        .range(range);
    return scale;
}

function prepare_plot(parent, owidth, oheight, title, x_axis, y_axis) {
    // Margins and size of the actual drawing area
    var m = {top: 10 , right: 10, bottom: 10, left: 10};
    if (title) { m.top += 40; }
    if (x_axis.range) { m.bottom += 20; }
    if (y_axis.range) {
        var extra_margin = 10 * Math.max(
            ('' + Math.floor(y_axis.range[0])).length,
            ('' + Math.ceil(y_axis.range[1])).length,
            3  // minimum extra margin
        );
        m.left += extra_margin;
    }
    if (x_axis.label) { m.bottom += 10; }
    if (y_axis.label) { m.left += 20; }
    var w = owidth - m.left - m.right;
    var h = oheight - m.top - m.bottom;

    // The x and y axes will be positioned relatively to *figure*
    var figure = parent.append('g')
        .attr('transform', 'translate(' + m.left + ', ' + m.top + ')');
    // The *plot* is where we actually want to draw something
    var plot = figure.append('g').attr({width: w, height: h});

    // Axes
    if (x_axis.range) {
        var x = scale(x_axis.range, [0, w]);
        var xa = figure.append('g')
            .attr('class', 'x axis')
            .attr('transform', 'translate(0, ' + h + ')')
            .call(d3.svg.axis()
                .scale(x)
                // .tickSize(0)
                // .tickPadding(6)
                .orient('bottom'));
        if (x_axis.unit) {
            xa.selectAll('.tick:nth-last-of-type(2) text').text(x_axis.unit);
        }
    }
    if (y_axis.range) {
        var y = scale(y_axis.range, [h, 0]);
        var ya = figure.append('g')
            .attr('class', 'y axis')
            .attr('transform', 'translate(0, 0)')
            .call(d3.svg.axis()
                .scale(y)
                .orient('left'));
        if (y_axis.unit) {
            ya.selectAll('.tick:nth-last-of-type(2) text').text(y_axis.unit);
        }
    }

    // Axis lables
    if (x_axis.label) {
        var xl = parent.append('text')
            .attr('class', 'axislabel')
            .attr('text-anchor', 'middle')
            .attr('x', m.left + w/2)
            .attr('y', m.top + h + m.bottom)
            .text(x_axis.label);
    }
    if (y_axis.label) {
        var yl = parent.append('text')
            .attr('class', 'axislabel')
            .text(y_axis.label);
        if (y_axis.label.length > 1) {
            yl.attr('text-anchor', 'middle')
              .attr('transform', 'rotate(-90)')
              .attr('x', - m.top - h/2)
              .attr('y', 0)
              .attr('dy', '1em');
        } else {
            yl.attr('text-anchor', 'left')
              .attr('x', 0)
              .attr('y', m.top + h/2);
        }
    }

    // Title
    var title_elem = null;
    if (title) {
        title_elem = parent.append('text')
            .attr('class', 'title')
            .attr('text-anchor', 'middle')
            .attr('x', owidth / 2)
            .attr('y', '1em')
            .text(title);
    }

    return {figure: figure, title: title_elem, plot: plot, w: w, h: h, m: m};
}
