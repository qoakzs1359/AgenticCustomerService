import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/user',
  },
  {
    path: '/user',
    name: 'UserChat',
    component: () => import('../views/UserChat.vue'),
    meta: { title: '用户聊天' },
  },
  {
    path: '/agent',
    name: 'AgentConsole',
    component: () => import('../views/AgentConsole.vue'),
    meta: { title: '客服后台' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
