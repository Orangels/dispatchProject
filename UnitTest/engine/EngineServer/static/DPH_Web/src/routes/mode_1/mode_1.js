import React from 'react';
import io from 'socket.io-client'
import video from 'video.js';
import videoSWF from 'videojs-swf/dist/video-js.swf';
import {Drawer, Col, Row, Tag, Button, Popover} from 'antd'
import {_fetch, get_2_float, deepCopy} from '../../utils/utils'
import Queue_len from '../../utils/dataStructure'
import "video.js/dist/video-js.css";
import back from '../../assets/back/back_large.jpg'
import bg from '../../assets/back/bg.jpg'
import './model_1.css'
import login_bg from "../../assets/bg/login_bg.jpg";

const uri = 'http://localhost/test';
const options = { transports: ['websocket'] };

const content_1_height = 600 - 230 + 150
// const content_1_height = 100
// const warning_img_height = 230
const warning_img_height = 150
const img_col = 5
const img_width = 250

// const url = window.location.origin
// const url = 'http://127.0.0.1:5000'
const url = 'http://192.168.88.221:5000'

class App extends React.Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            src:[],
            persons: new Queue_len(8),
            // drawer
            visible: false,
            cpu_percent:0,
            cpu_temp:0,
            gpu_percent:0,
            memory_total:'0M',
            memory_used:'0M',
            disk_total:'0G',
            disk_used:'0G',
            memory_percent:0,
            disk_percent:0
        };

        this._ws_new_state = this._ws_new_state.bind(this)
        this.waring_img_history = this.waring_img_history.bind(this)
        this.socket = 1
    }

    _ws_new_state(data){

        let results = data.result
        console.log('*******')
        console.log(results)
        let persons = this.state.persons
        console.log(persons)
        persons.push(results, true)
        this.setState({
            persons:persons
        })



        // this.setState({
        //     src:`data:image/png;base64,${data.data.img}`
        //     // src:`http://192.168.88.91:9000${data.data.img}`
        // },()=>{
        // })

    }

    componentDidMount() {
        //拉流
        let options_1 = {
            autoplay:    true,
            controls:    true,
            preload:     true, //预加载
            fluid:       false, //播放器将具有流畅的大小。换句话说，它将扩展以适应其容器
            // aspectRatio: '16:9',//将播放器置于流体模式，在计算播放器的动态大小时使用。由冒号（"16:9"或"4:3"）分隔的两个数字
            techOrder:   ['html5'],//Video.js技术首选的顺序
            live: true,
            sources: [{
                type: "application/x-mpegURL",
                // src: "http://192.168.88.27/hls/room_1.m3u8",
                // src: "http://192.168.88.92:8080/hls/test.m3u8",
                // src: "http://192.168.88.25/hls/room.m3u8",
                src: `http://${window.location.hostname}/hls/room.m3u8`,
                withCredentials: false
            }],
            // html5: { hls: { withCredentials: true } },
            html5: { hls: { withCredentials: false } },
        }

        this.player_1 = video('example_video_1',options_1);


        let url_socket = `${url}/Camera_Web_ws`

        //本机测试 用固定 url
        console.log('长连接 服务器')
        this.socket = io(url_socket)
        this.socket.on('new_state',this._ws_new_state);
        this.start_time = new Date().getTime()
        // this.socket.on('new_person_state',this._ws_new_person_state);
    }

    componentWillUnmount() {
        this.player_1.dispose()
        this.socket.emit('disconnect')
        this.timer && clearInterval(this.timer)
    }

    waring_img_history = (persons_arr, length)=>{
        //倒叙
        let persons_arr_cp = deepCopy(persons_arr)
        let persons = persons_arr_cp.data
        persons.reverse()
        let persons_history = persons.map((person, i)=>{
            console.log(person)
            let color = person.rec ? "cyan" : "red"
            return (
                <Col span={24/length} style={{position:'relative',}}>
                    <Tag color={color} style={styles.waring_tag} >
                        {person.name}
                    </Tag>
                    <Tag color={color} style={{position: 'absolute',
                        top:10,
                        right:20}} >
                        {person.date}
                    </Tag>
                    <img width={"100%"} height={90} src={`${url}${person.img}`} style={{}}/>
                </Col>
            )
        })
        return persons_history
    }

    render() {

        return(
            <div className="Mode_1" style={styles.wrap_div}>
                <Row gutter={16} style={{ padding:10}}>
                    {/*{this.waring_img_history([*/}
                    {/*    {name:"刘森", img:back, rec:true},*/}
                    {/*    {name:"刘森", img:back, rec:true},*/}
                    {/*    {name:"刘森", img:back, rec:false},*/}
                    {/*], 8)}*/}
                    {
                        this.waring_img_history(this.state.persons, 8)
                    }
                </Row>
                <Row gutter={16}
                     // style={{backgroundColor:'#F0F2F5', padding:10}}
                     style={{ padding:10}}
                >
                    <Col span={24} >
                        <Tag color={'#FA0F21'} style={{position: 'absolute', top:10, right:10, zIndex:99}}>
                            {'鱼眼全屏'}
                        </Tag>
                        <video id="example_video_1" className="video-js vjs-custom-skin"  preload="auto"  data-setup=''
                               style={{width:'100%', height:content_1_height}}
                               ref={(input) => { this.video = input; }}

                        >
                            <source src="rtmp://127.0.0.1:1935/rtmplive/room" type="rtmp/flv"/>
                        </video>
                    </Col>
                </Row>
            </div>
        )
    }

}

const styles = {
    wrap_div:{
        background:`url(${bg}) no-repeat `,
        backgroundSize: '100% 100%',
        transition:'all .5s'
    },
    waring_img :{
        width:img_width,
        height:warning_img_height,
        position:'relative',
        marginTop:20
    },
    waring_tag:{
        position: 'absolute',
        top:10,
        left:20
    },
    device_div:{
        display:"flex",
        flexDirection:"row",
        width: 180,
    },
    device_span:{
        color:'#FFFFFF'
    },
    device_tag:{
        marginLeft:10,
    }
}

export default App;
