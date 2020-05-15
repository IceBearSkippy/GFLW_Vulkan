#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <vector>
#include <vulkan/vulkan.h>
#include <fstream>
#include <set>
#include <optional>
#include <algorithm>

using namespace std;

const vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
    optional<uint32_t> graphicsFamily;
    optional<uint32_t> presentFamily;
    bool isComplete();
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    vector<VkSurfaceFormatKHR> formats;
    vector<VkPresentModeKHR> presentModes;
};

namespace Utils {
    vector<char> readFile(const string& filename);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR& surface);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR& surface);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    bool isDeviceSuitible(VkPhysicalDevice device, VkSurfaceKHR& surface);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t WIDTH, uint32_t HEIGHT);
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const
        VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const
        VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT*
        pDebugMessenger);
    void DestroyDebugUtilsMessengerEXT(VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger, const
        VkAllocationCallbacks* pAllocator);
}
