import React from 'react';
import io from 'socket.io-client'
import video from 'video.js';
import videoSWF from 'videojs-swf/dist/video-js.swf';
import "video.js/dist/video-js.css";
import "videojs-flash"

import { Button } from "antd";
import {_fetch} from '../../utils/utils'

const uri = 'http://localhost/test';
const options = { transports: ['websocket'] };


class App extends React.Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            src:'',
            disable:false,
        };
        this._ws_new_state = this._ws_new_state.bind(this)
        this.socket = 1


    }

    url = 'http://127.0.0.1:7000/set_imgs'
    ws_url = 'http://127.0.0.1:7000/Camera_Web_ws'

    _ws_new_state(data){
        let start = new Date().getTime()
        this.setState({
            src:`data:image/png;base64,${data.data.img}`
        },()=>{
            let end =new Date().getTime()
            // console.log(`${end} - ${start} = ${end-start}`)
            console.log(`${end} - ${this.start_time} = ${end-this.start_time}`)
            this.start_time = end
        })
    }

    componentDidMount() {

        // let options = {
        //     autoplay:    true,
        //     controls:    false,
        //     preload:     true, //预加载
        //     fluid:       true, //播放器将具有流畅的大小。换句话说，它将扩展以适应其容器
        //     // poster:      videoPNG,//播放前显示的视频画面，播放开始之后自动移除。通常传入一个URL
        //     techOrder:   ['flash'],//Video.js技术首选的顺序
        //     aspectRatio: '16:9',//将播放器置于流体模式，在计算播放器的动态大小时使用。由冒号（"16:9"或"4:3"）分隔的两个数字
        //     notSupportedMessage: '此视频暂无法播放，请稍后再试', //允许覆盖Video.js无法播放媒体源时显示的默认信息
        //     sources: [{
        //         type: "rtmp/flv",
        //         src: 'rtmp://127.0.0.1:1935/hls/room'
        //     }],
        //     flash: { swf: videoSWF },
        //     live: true,
        // }


        let options = {
            autoplay:    true,
            controls:    true,
            preload:     true, //预加载
            fluid:       true, //播放器将具有流畅的大小。换句话说，它将扩展以适应其容器
            aspectRatio: '16:9',//将播放器置于流体模式，在计算播放器的动态大小时使用。由冒号（"16:9"或"4:3"）分隔的两个数字
            techOrder:   ['html5'],//Video.js技术首选的顺序
            live: true,
            sources: [{
                type: "application/x-mpegURL",
                // src: "http://192.168.88.27/hls/room_1.m3u8",
                // src: "http://192.168.88.92:8080/hls/test.m3u8",
                src: "http://192.168.88.57:8080/hls/room.m3u8",
                withCredentials: false
            }],
            // html5: { hls: { withCredentials: true } },
            html5: { hls: { withCredentials: false } },
        }


        this.player = video('example_video_1',options);


        // let url = window.location.origin;
        let url = '192.168.88.91:7000/'
        url = `${url}Camera_Web_ws`
        // let socket = io(url);

        //本机测试 用固定 url
        // console.log('长连接 服务器')
        // this.socket = io(url)
        // this.socket.on('new_state',this._ws_new_state);
        // this.start_time = new Date().getTime()
        // this.socket.on('new_person_state',this._ws_new_person_state);

    }

    componentWillUnmount() {
        this.player.dispose()
    }


    render() {
        let content_1_height = 500
        return(
            <div className="Mode_2">
                <video id="example_video_1" className="video-js vjs-custom-skin vjs-fluid"  preload="auto"  data-setup=''
                       height={content_1_height}
                    ref={(input) => { this.video = input; }}

                >
                    <source src="rtmp://127.0.0.1:1935/rtmplive/room" type="rtmp/flv"/>
                </video>
            </div>
        )
    }

}



export default App;
