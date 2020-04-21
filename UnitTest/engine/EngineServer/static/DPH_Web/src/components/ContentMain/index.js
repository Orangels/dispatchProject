import React from 'react'
import { withRouter, Switch, Redirect } from 'react-router-dom'
import LoadableComponent from '../../utils/LoadableComponent'
import PrivateRoute from '../PrivateRoute'

const Home = LoadableComponent(()=>import('../../routes/Home/index'))  //参数一定要是函数，否则不会懒加载，只会代码拆分

//基本组件Demo
const ButtonDemo = LoadableComponent(()=>import('../../routes/General/ButtonDemo/index'))
const IconDemo = LoadableComponent(()=>import('../../routes/General/IconDemo/index'))

//导航组件Demo
const DropdownDemo = LoadableComponent(()=>import('../../routes/Navigation/DropdownDemo/index'))
const MenuDemo = LoadableComponent(()=>import('../../routes/Navigation/MenuDemo/index'))
const StepsDemo = LoadableComponent(()=>import('../../routes/Navigation/StepsDemo/index'))

//输入组件Demo
const FormDemo1 = LoadableComponent(()=>import('../../routes/Entry/FormDemo/FormDemo1'))
const FormDemo2 = LoadableComponent(()=>import('../../routes/Entry/FormDemo/FormDemo2'))
const UploadDemo = LoadableComponent(()=>import('../../routes/Entry/UploadDemo/index'))

//显示组件Demo
const CarouselDemo = LoadableComponent(()=>import('../../routes/Display/CarouselDemo/index'))
const CollapseDemo = LoadableComponent(()=>import('../../routes/Display/CollapseDemo/index'))
const ListDemo = LoadableComponent(()=>import('../../routes/Display/ListDemo/index'))
const TableDemo = LoadableComponent(()=>import('../../routes/Display/TableDemo/index'))
const TabsDemo = LoadableComponent(()=>import('../../routes/Display/TabsDemo/index'))

//反馈组件Demo
const SpinDemo = LoadableComponent(()=>import('../../routes/Feedback/SpinDemo/index'))
const ModalDemo = LoadableComponent(()=>import('../../routes/Feedback/ModalDemo/index'))
const NotificationDemo = LoadableComponent(()=>import('../../routes/Feedback/NotificationDemo/index'))

//其它
const AnimationDemo = LoadableComponent(()=>import('../../routes/Other/AnimationDemo/index'))
const GalleryDemo = LoadableComponent(()=>import('../../routes/Other/GalleryDemo/index'))
const DraftDemo = LoadableComponent(()=>import('../../routes/Other/DraftDemo/index'))
const ChartDemo = LoadableComponent(()=>import('../../routes/Other/ChartDemo/index'))
const LoadingDemo = LoadableComponent(()=>import('../../routes/Other/LoadingDemo/index'))
const ErrorPage = LoadableComponent(()=>import('../../routes/Other/ErrorPage/index'))
const SpringText = LoadableComponent(()=>import('../../routes/Other/SpringText/index'))

//关于
const About = LoadableComponent(()=>import('../../routes/About/index'))

//服务器管理

//拉流模式
const Mode_1_com = LoadableComponent(()=>import('../../routes/mode_1/mode_1'))
const Mode_2_com = LoadableComponent(()=>import('../../routes/mode_2/mode_2'))


@withRouter
class ContentMain extends React.Component {
  render () {
    return (
      <div
          style={{padding: 16, position: 'relative'}}
      >
        <Switch>
          {/*<PrivateRoute exact path='/home' component={Home}/>*/}
          <PrivateRoute exact path='/mode_1' component={Mode_1_com}/>
          <PrivateRoute exact path='/mode_2' component={Mode_2_com}/>
          <PrivateRoute exact path='/home/config' component={About}/>
          <PrivateRoute exact path='/home/about' component={About}/>
          <Redirect exact from='/' to='/mode_1'/>
        </Switch>
      </div>
    )
  }
}

export default ContentMain