import React from 'react'
import {Layout} from 'antd'
import SiderNav from '../../components/SiderNav'
import ContentMain from '../../components/ContentMain'
import HeaderBar from '../../components/HeaderBar'

import bg from '../../assets/back/bg.jpg'
import header from '../../assets/back/hader.png'

const {Sider, Header, Content, Footer} = Layout


class Index extends React.Component{
  state = {
    // collapsed: false
    collapsed: true
  }

  toggle = () => {
    // console.log(this)  状态提升后，到底是谁调用的它
    this.setState({
      collapsed: !this.state.collapsed
    })
  }
  render() {
    // 设置Sider的minHeight可以使左右自适应对齐
    return (
      <div id='page'style={{
        background:`url(${bg}) no-repeat `,
        backgroundSize: '100% 100%',
      }}>
        <Layout >
          <Sider collapsible
                 trigger={null}
                 collapsed={this.state.collapsed}
                 >
            <SiderNav/>
          </Sider>
          <Layout style={{
            background:`url(${bg}) no-repeat `,
            backgroundSize: '100% 100%',
          }}>
            <Header
                // style={{background: '#fff', padding: '0 16px'}}
                style={{
                  background:`url(${header}) no-repeat `,
                  backgroundSize: '100% 100%',
                }}
            >
              <HeaderBar collapsed={this.state.collapsed} onToggle={this.toggle}/>
              <span
                  style={{left: "49%", top:0,  position: 'absolute', color:'white', zIndex:99, fontSize:24}}
              >
                            ATM 告警
              </span>
            </Header>
            <Content>
              <ContentMain/>
            </Content>
            <Footer
                // style={{textAlign: 'center', padding: '12px 50px'}}
                style={{
                  background:`url(${bg}) no-repeat `,
                  backgroundSize: '100% 100%',
                  textAlign: 'center', padding: '5px 50px',
                  color:'#FFFFFF'
                }}
            >©2020 Created by Priv </Footer>
          </Layout>
        </Layout>
      </div>
    );
  }
}
export default Index